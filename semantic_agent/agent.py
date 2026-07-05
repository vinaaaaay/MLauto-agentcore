"""
Single build agent compiling the Semantic Agent StateGraph with inline nodes.
"""

import os
import asyncio
import logging
import time
import uuid
import json
import boto3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from .utils import SemanticAgentState, TutorialInfo, AgentCoreVectorStoreClient
from .prompts import _QUERY_GENERATOR_PROMPT, _RERANKER_PROMPT

logger = logging.getLogger(__name__)

from common_local.metrics_context import MetricsContext
from common_local.metrics_emitter import node_metrics, emit_event

metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="semantic_agent")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")


def build_semantic_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Semantic Agent LangGraph.
    
    Contains all graph nodes inline to encapsulate execution scope,
    matching the single build agent design pattern.
    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")

    
    def _init_llm(llm_config: dict):
        """Helper to initialize ChatOpenAI or ChatOpenRouter directly from config."""
        model = llm_config.get("model", "gpt-4o")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens") # None means use default API limit

        is_openai = model.lower().startswith("gpt") or model.lower().startswith("o1-") or model.lower().startswith("o3-")

        if is_openai:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Export it before running: export OPENAI_API_KEY=sk-..."
                )

            is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

            if is_reasoning_model:
                logger.info("Detected reasoning model. Forcing temp=1.")
                kwargs = {"model": model, "temperature": 1, "api_key": api_key, "max_retries": 1, "timeout": 60.0}
                if max_tokens is not None:
                    kwargs["max_completion_tokens"] = max_tokens
                return ChatOpenAI(**kwargs)
            
            logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
            kwargs = {"model": model, "temperature": temperature, "api_key": api_key, "max_retries": 1, "timeout": 60.0}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenAI(**kwargs)

        else:
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY environment variable is not set."
                )
            
            logger.info(f"Initialized OpenRouter via ChatOpenAI: model={model}, temp={temperature}")
            kwargs = {
                "model": model,
                "temperature": temperature,
                "api_key": openrouter_api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "max_retries": 1,
                "timeout": 60.0
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenAI(**kwargs)

    async def _mcp_tool_call(server_url: str, query: str, tool_name: str, top_k: int, condensed: bool, callbacks: list) -> List[Dict[str, Any]]:
        """
        Instrumented and authenticated wrapper around the vector store MCP server retrieval.
        Emits tool_call events matching SessionMetricsCallback format.
        
        Made async so it can be awaited from both ainvoke (A2A path) and
        wrapped with asyncio.run() from a sync context — avoids RuntimeError
        from calling asyncio.run() inside an already-running event loop.
        """
        run_id = str(uuid.uuid4())
        parent_run_id = str(uuid.uuid4())
        tool_input = json.dumps({
            "query": query,
            "tool_name": tool_name,
            "top_k": top_k,
            "condensed": condensed,
        }, default=str)
        t0 = time.time()

        status = "success"
        error_str = None
        raw_tutorials = []
        try:
            logger.info(f"Using AgentCoreVectorStoreClient for ARN: {server_url}")
            agentcore_client = AgentCoreVectorStoreClient(
                agent_runtime_arn=server_url,
                region=AWS_REGION,
            )
            raw_tutorials = await agentcore_client.retrieve_tutorials(
                query=query,
                tool_name=tool_name,
                top_k=top_k,
                condensed=condensed,
                callbacks=callbacks,
            )
        except Exception as e:
            status = "error"
            error_str = str(e)
            raise
        finally:
            latency = (time.time() - t0) * 1000
            event = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
                "event_type": "tool_call",
                **(active_ctx.snapshot() if active_ctx else {}),
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "node_name": "retrieve_tutorials",
                "tool_name": "mcp_retrieve_tutorials",
                "tool_input": tool_input,
                "latency_ms": round(latency, 2),
                "status": status,
            }
            if status == "success":
                event["tool_output"] = json.dumps({"tutorial_count": len(raw_tutorials)}, default=str)
            else:
                event["error"] = error_str
            if active_logger:
                active_logger.info(json.dumps(event))

        return raw_tutorials

    @node_metrics(active_ctx, active_logger, "generate_query")
    def generate_query(state: SemanticAgentState, config: dict = None) -> dict:
        """LLM node to generate a search query from the agent state."""
        logger.info("─── [Semantic Agent] generate_query ───")

        agent_config = state.get("config", {})
        llm_config = agent_config.get("llm", {}).copy()
        
        # Default to 'gpt-4o-mini' if not specified
        if "model" not in llm_config:
            llm_config["model"] = "gpt-4o-mini"

        try:
            llm = _init_llm(llm_config)
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return {"search_query": state.get("task_description", "")[:256]}

        # Format the prompt
        task_desc = state.get("task_description", "")
        data_prompt = state.get("data_prompt", "")
        user_input = state.get("user_input", "")
        all_error_analyses = "\n\n".join(state.get("all_error_analyses", [])) or "None"
        selected_tool = state.get("current_tool", "")

        prompt = _QUERY_GENERATOR_PROMPT.format(
            task_description=task_desc,
            data_prompt=data_prompt,
            user_input=user_input,
            all_previous_error_analyses=all_error_analyses,
            selected_tool=selected_tool,
        )

        try:
            response = llm.invoke(prompt)
            search_query = response.content.strip().strip("\"'")
            if active_ctx and active_logger:
                emit_event(active_logger, {
                    **active_ctx.snapshot(),
                    "event_type": "debug",
                    "node_name": "generate_query",
                    "prompt": prompt,
                    "response": search_query,
                    "prompt_len": len(prompt),
                    "response_len": len(search_query),
                })
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            search_query = ""

        # Clean up prefixes from LLM response
        for prefix in ["search query:", "query:", "the search query is:"]:
            if search_query.lower().startswith(prefix):
                search_query = search_query[len(prefix):].strip()
                break

        if not search_query:
            search_query = (task_desc or selected_tool)[:256]
            logger.warning("Failed to generate query from LLM; using fallback.")

        logger.info(f"Generated search query: '{search_query}'")
        return {"search_query": search_query}

    @node_metrics(active_ctx, active_logger, "retrieve_tutorials")
    async def retrieve_tutorials(state: SemanticAgentState, config: dict = None) -> dict:
        """MCP client node — calls the standalone Vector Store MCP server using the client wrapper."""
        logger.info("─── [Semantic Agent] retrieve_tutorials ───")

        agent_config = state.get("config", {})
        mcp_servers = agent_config.get("mcp_servers", {})
        server_url = mcp_servers.get("vector_store_url", "http://localhost:8010")

        tutorials_config = agent_config.get("tutorials", {})
        top_k = tutorials_config.get("num_tutorial_retrievals", 5)
        condensed = tutorials_config.get("condense_tutorials", False)

        query = state.get("search_query", "")
        tool_name = state.get("current_tool", "")

        # Extract LangGraph callbacks so MCP client can trigger on_tool_start/on_tool_end
        callbacks = config.get("callbacks", []) if config else []

        mcp_calls = []
        try:
            t0 = time.time()
            raw_tutorials = await _mcp_tool_call(
                server_url=server_url,
                query=query,
                tool_name=tool_name,
                top_k=top_k,
                condensed=condensed,
                callbacks=callbacks,
            )
            elapsed = time.time() - t0
            
            mcp_output = {
                "results": raw_tutorials,
                "result_count": len(raw_tutorials),
                "query": query,
                "tool_name": tool_name
            }
            
            call_log = {
                "timestamp": datetime.now().isoformat(),
                "skill": "mcp_retrieve_tutorials",
                "input": {
                    "query": query,
                    "tool_name": tool_name,
                    "top_k": top_k,
                    "condensed": condensed
                },
                "output": mcp_output,
                "time_taken_seconds": round(elapsed, 3)
            }
            mcp_calls.append(call_log)
        except Exception as e:
            elapsed = time.time() - t0
            call_log = {
                "timestamp": datetime.now().isoformat(),
                "skill": "mcp_retrieve_tutorials",
                "input": {
                    "query": query,
                    "tool_name": tool_name,
                    "top_k": top_k,
                    "condensed": condensed
                },
                "output": None,
                "error": str(e),
                "time_taken_seconds": round(elapsed, 3)
            }
            mcp_calls.append(call_log)
            logger.error(f"Vector Store MCP retrieval call failed: {e}")
            raw_tutorials = []

        # Deserialize to TutorialInfo named tuples
        tutorials = []
        for t in raw_tutorials:
            try:
                tutorials.append(TutorialInfo(
                    path=Path(t["path"]),
                    title=t["title"],
                    summary=t.get("summary", ""),
                    score=t.get("score"),
                    content=t.get("content"),
                ))
            except Exception as e:
                logger.warning(f"Failed to deserialize tutorial: {e}")

        logger.info(f"Retrieved {len(tutorials)} tutorial candidates from MCP server")
        return {"tutorial_retrieval": tutorials, "mcp_calls": mcp_calls}

    @node_metrics(active_ctx, active_logger, "rerank_tutorials")
    def rerank_tutorials(state: SemanticAgentState, config: dict = None) -> dict:
        """Reranks the retrieved tutorials locally using LLM-based selection."""
        logger.info("─── [Semantic Agent] rerank_tutorials (running local/in-process reranker) ───")

        agent_config = state.get("config", {})
        tutorials_config = agent_config.get("tutorials", {})
        
        max_num = tutorials_config.get("max_num_tutorials", 3)
        max_length = tutorials_config.get("max_tutorial_length", 30000)
        use_summary = tutorials_config.get("use_tutorial_summary", True)

        tutorials = state.get("tutorial_retrieval", [])
        if not tutorials:
            logger.warning("  No tutorials to rerank")
            return {"tutorial_prompt": ""}

        # 1. Format tutorials info for the LLM selection prompt
        tutorials_info_lines = []
        for i, tutorial in enumerate(tutorials):
            summary_text = getattr(tutorial, "summary", "") if use_summary else ""
            summary_text = summary_text or "(No summary available)"
            tutorials_info_lines.append(
                f"{i + 1}. Title: {getattr(tutorial, 'title', 'Untitled')}\n   Summary: {summary_text}"
            )
        tutorials_info = "\n".join(tutorials_info_lines)

        all_error_analyses = state.get("all_error_analyses", [])
        all_errors = "\n\n".join(all_error_analyses) or "None"

        # Format the selection prompt
        prompt = _RERANKER_PROMPT.format(
            task_description=state.get("task_description", ""),
            data_prompt=state.get("data_prompt", ""),
            user_input=state.get("user_input", ""),
            all_previous_error_analyses=all_errors,
            tutorials_info=tutorials_info,
            max_num_tutorials=max_num,
        )

        try:
            # LLM setup — SessionMetricsCallback (passed via LangGraph config) auto-captures the llm_call event
            llm_config = agent_config.get("llm", {}).copy()
            if "model" not in llm_config:
                llm_config["model"] = "gpt-4o-mini"
            llm = _init_llm(llm_config)

            response = llm.invoke(prompt)
            response_text = response.content

            if active_ctx and active_logger:
                emit_event(active_logger, {
                    **active_ctx.snapshot(),
                    "event_type": "debug",
                    "node_name": "rerank_tutorials",
                    "prompt": prompt,
                    "response": response_text,
                    "prompt_len": len(prompt),
                    "response_len": len(response_text),
                })

            response = response_text

            # Parse response
            content_line = response.strip().split("\n")[0]
            content_clean = "".join(c for c in content_line if c.isdigit() or c == ",")

            selected_tutorials = []
            if content_clean:
                try:
                    indices = [int(idx.strip()) - 1 for idx in content_clean.split(",") if idx.strip()]
                    for idx in indices:
                        if 0 <= idx < len(tutorials):
                            selected_tutorials.append(tutorials[idx])
                except ValueError as e:
                    logger.warning(f"  Error parsing tutorial indices: {e}")
            
            # Fallback to top-k by retrieval score if parsing failed or returned empty
            if not selected_tutorials:
                logger.warning("  Reranking failed; falling back to top tutorials by score")
                sorted_tutorials = sorted(tutorials, key=lambda t: getattr(t, "score", 0.0) or 0.0, reverse=True)
                selected_tutorials = sorted_tutorials[:max_num]
            else:
                selected_tutorials = selected_tutorials[:max_num]

            # Format selected tutorials content into the prompt
            per_tutorial_length = max_length // max(1, len(selected_tutorials))
            formatted_parts = []

            for tutorial in selected_tutorials:
                content = getattr(tutorial, "content", "")
                if not content:
                    logger.warning(f"  Tutorial '{getattr(tutorial, 'title', 'Untitled')}' has no content — skipping")
                    continue
                if len(content) > per_tutorial_length:
                    content = content[:per_tutorial_length] + "\n...(truncated)"
                formatted_parts.append(f"### {getattr(tutorial, 'title', 'Untitled')}\n{content}")

            tutorial_prompt = "\n\n".join(formatted_parts) if formatted_parts else ""
            logger.info(f"  Selected {len(selected_tutorials)} tutorials. tutorial_prompt length: {len(tutorial_prompt)} chars")
            return {"tutorial_prompt": tutorial_prompt}

        except Exception as e:
            logger.error(f"In-process reranking failed: {e}")
            # Fallback in case of LLM or connection failure
            sorted_tutorials = sorted(tutorials, key=lambda t: getattr(t, "score", 0.0) or 0.0, reverse=True)
            selected_tutorials = sorted_tutorials[:max_num]
            formatted_parts = []
            for tutorial in selected_tutorials:
                content = getattr(tutorial, "content", "")
                if content:
                    formatted_parts.append(f"### {getattr(tutorial, 'title', 'Untitled')}\n{content[:max_length // max_num]}")
            return {"tutorial_prompt": "\n\n".join(formatted_parts)}

    # Graph Setup
    graph = StateGraph(SemanticAgentState)
    graph.add_node("generate_query", generate_query)
    graph.add_node("retrieve_tutorials", retrieve_tutorials)
    graph.add_node("rerank_tutorials", rerank_tutorials)

    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "retrieve_tutorials")
    graph.add_edge("retrieve_tutorials", "rerank_tutorials")
    graph.add_edge("rerank_tutorials", END)

    return graph.compile()
