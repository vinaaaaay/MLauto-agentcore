"""
Helper utilities and abstractions for the Semantic Agent.
Keeps the agent self-contained and independent.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict, Any, TypedDict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════

class TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class SemanticAgentState(TypedDict):
    """
    State representing the data flow through the Semantic Agent graph.
    Supports a flexible dictionary model to easily support additional parameters.
    """
    # Configuration
    config: Dict[str, Any]
    output_folder: str

    # Context & Inputs
    task_description: str
    data_prompt: str
    user_input: str
    all_error_analyses: List[str]
    current_tool: str

    # Outputs
    search_query: str
    tutorial_retrieval: List[TutorialInfo]
    tutorial_prompt: str


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Call Logger
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCallLogger:
    """Logs every LLM call (prompt + response) to structured JSONL."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        self.call_count += 1
        call_id = self.call_count

        logger.info(f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)")

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )

        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "prompt_length": len(prompt),
            "response_length": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "response": content,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM call log: {e}")

        return content


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Client
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  AgentCore MCP Client (ARN-based invocation)
# ═══════════════════════════════════════════════════════════════════════════

class AgentCoreVectorStoreClient:
    """
    Invokes the Vector Store MCP server deployed on Bedrock AgentCore
    by ARN via the bedrock-agentcore-runtime boto3 client.

    Use this client when VECTOR_STORE_URL is an AgentCore agent runtime ARN
    (i.e. starts with 'arn:aws:').  The mcpserver app.py entrypoint returns
    a dict with a 'results' key containing the list of tutorials.
    """

    def __init__(self, agent_runtime_arn: str, region: str = "ap-south-1"):
        import boto3
        from botocore.config import Config
        self.agent_runtime_arn = agent_runtime_arn
        
        # Configure client with increased read timeout and no retries
        config = Config(
            read_timeout=300,
            connect_timeout=60,
            retries={'max_attempts': 1}
        )
        self.client = boto3.client("bedrock-agentcore", region_name=region, config=config)

    async def retrieve_tutorials(
        self,
        query: str,
        tool_name: str,
        top_k: int = 5,
        condensed: bool = False,
        callbacks: list = None,
    ) -> List[Dict[str, Any]]:
        """
        Invoke the mcpserver AgentCore runtime and return the tutorials list.
        Callback hooks (on_tool_start / on_tool_end) are fired around the call.
        """
        import uuid
        import asyncio

        run_id = uuid.uuid4()
        parent_run_id = uuid.uuid4()

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_tool_start"):
                    try:
                        cb.on_tool_start(
                            {"name": "agentcore_retrieve_tutorials"},
                            json.dumps({"query": query, "tool_name": tool_name,
                                        "top_k": top_k, "condensed": condensed}),
                            run_id=run_id,
                            parent_run_id=parent_run_id,
                        )
                    except Exception as cb_err:
                        logger.warning(f"on_tool_start callback failed: {cb_err}")

        try:
            payload = {"query": query, "tool_name": tool_name, "top_k": top_k, "condensed": condensed}
            # boto3 calls are synchronous — run in a thread pool to avoid blocking the event loop
            raw = await asyncio.get_event_loop().run_in_executor(
                None, self._invoke_sync, payload
            )

            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_tool_end"):
                        try:
                            cb.on_tool_end(
                                json.dumps({"tutorial_count": len(raw)}),
                                run_id=run_id,
                                parent_run_id=parent_run_id,
                            )
                        except Exception as cb_err:
                            logger.warning(f"on_tool_end callback failed: {cb_err}")
            return raw

        except Exception as e:
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_tool_error"):
                        try:
                            cb.on_tool_error(e, run_id=run_id, parent_run_id=parent_run_id)
                        except Exception as cb_err:
                            logger.warning(f"on_tool_error callback failed: {cb_err}")
            raise

    def _invoke_sync(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synchronous AgentCore invocation.  Called via run_in_executor so it
        doesn't block the asyncio event loop.

        The mcpserver @app.entrypoint returns:
            {"results": [...], "result_count": int, "query": str, "tool_name": str}
        """
        logger.info(
            f"[AgentCoreVectorStoreClient] Invoking ARN={self.agent_runtime_arn!r} "
            f"payload={json.dumps({k: v for k, v in payload.items() if k != 'query'})}"
        )
        response = self.client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_runtime_arn,
            qualifier="DEFAULT",
            contentType="application/json",
            accept="application/json",
            payload=json.dumps(payload).encode()
        )
        body = response.get("response") or response.get("body")
        if hasattr(body, "read"):
            body_bytes = body.read()
        else:
            body_bytes = body
        text = body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else str(body_bytes)
        data = json.loads(text)
        logger.info(f"[AgentCoreVectorStoreClient] Response keys: {list(data.keys())}")

        # mcpserver returns {"results": [...], ...}
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        # Fallback: if the response is already a list
        if isinstance(data, list):
            return data
        logger.warning(f"[AgentCoreVectorStoreClient] Unexpected response format: {str(data)[:200]}")
        return []
