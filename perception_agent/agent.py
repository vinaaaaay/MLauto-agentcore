"""
Perception Agent — LangGraph StateGraph with inline nodes.

Graph flow:
  START → scan_data → find_description_files → generate_task_description
        → select_tools → END

Nodes:
  1. scan_data              — List sandbox files, group by pattern, read via LLM
  2. find_description_files — Identify README/description files using LLM
  3. generate_task_description — Synthesize concise ML task description
  4. select_tools           — Rank available ML libraries for the task

This agent is completely self-contained — deployable as a standalone
Bedrock AgentCore service.
"""

import difflib
import logging
import os
import random
import re
import json
import time
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

try:
    # Package import (used by app.py: from perception_agent.agent import ...)
    from .state import PerceptionAgentState
    from .prompts import (
        PYTHON_READER_PROMPT,
        DESCRIPTION_FILE_RETRIEVER_PROMPT,
        TASK_DESCRIPTOR_PROMPT,
        TOOL_SELECTOR_PROMPT,
    )
    from .utils import (
        _get_llm,
        _get_sandbox_client,
        _get_all_files_sandbox,
        _group_similar_files,
        _pattern_to_path,
        _extract_code,
        _execute_code_sandbox,
        _ToolRegistry,
        MAX_CHARS_PER_FILE,
        MAX_FILE_GROUP_SIZE_TO_SHOW,
        NUM_EXAMPLE_FILES_TO_SHOW,
    )
    from .common_local.metrics_context import MetricsContext
    from .common_local.metrics_emitter import node_metrics, emit_event
except ImportError:
    # Standalone / direct execution
    from state import PerceptionAgentState
    from prompts import (
        PYTHON_READER_PROMPT,
        DESCRIPTION_FILE_RETRIEVER_PROMPT,
        TASK_DESCRIPTOR_PROMPT,
        TOOL_SELECTOR_PROMPT,
    )
    from utils import (
        _get_llm,
        _get_sandbox_client,
        _get_all_files_sandbox,
        _group_similar_files,
        _pattern_to_path,
        _extract_code,
        _execute_code_sandbox,
        _ToolRegistry,
        MAX_CHARS_PER_FILE,
        MAX_FILE_GROUP_SIZE_TO_SHOW,
        NUM_EXAMPLE_FILES_TO_SHOW,
    )
    from common_local.metrics_context import MetricsContext
    from common_local.metrics_emitter import node_metrics, emit_event

logger = logging.getLogger(__name__)

metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="perception_agent")


def build_perception_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Perception Agent LangGraph.

    All graph nodes are defined inline to encapsulate execution scope,
    matching the single build agent design pattern used across FAME.
    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")

    # ─── Helper ──────────────────────────────────────────────────────────

    def _sandbox_tool_call(sandbox, tool_name: str, node_name: str, **kwargs):
        """
        Instrumented wrapper around sandbox operations.
        Emits tool_call events for observability.
        """
        run_id = str(uuid.uuid4())
        parent_run_id = str(uuid.uuid4())
        tool_input = json.dumps(kwargs, default=str)
        t0 = time.time()

        status = "success"
        output = ""
        error_str = None
        result = None
        try:
            if tool_name == "sandbox_write_file":
                result = sandbox.write_file_sync(kwargs["path"], kwargs["content"])
                output = json.dumps({"path": kwargs["path"], "bytes_written": len(kwargs.get("content", ""))}, default=str)
            elif tool_name == "sandbox_read_file":
                result = sandbox.read_file_sync(kwargs["path"])
                output = json.dumps({"path": kwargs["path"], "bytes_read": len(result) if result else 0}, default=str)
            elif tool_name == "sandbox_exec_shell":
                success, stdout, stderr = sandbox.exec_shell_sync(
                    command=kwargs["command"],
                    cwd=kwargs.get("cwd", "/home/gem/workspace"),
                )
                result = (success, stdout, stderr)
                output = json.dumps({
                    "success": success,
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                }, default=str)
            elif tool_name == "sandbox_exec_python":
                success, stdout, stderr = _execute_code_sandbox(
                    kwargs["code"], language="python", sandbox=sandbox,
                    timeout=kwargs.get("timeout", 60),
                )
                result = (success, stdout, stderr)
                output = json.dumps({
                    "success": success,
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                }, default=str)
            else:
                raise ValueError(f"Unknown sandbox tool: {tool_name}")
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
                "node_name": node_name,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "latency_ms": round(latency, 2),
                "status": status,
            }
            if status == "success":
                event["tool_output"] = output
            else:
                event["error"] = error_str
            if active_logger:
                active_logger.info(json.dumps(event))

        return result

    def _read_file_via_llm(llm, file_path: str, max_chars: int, file_size: int, sandbox, node_name: str) -> str:
        """
        Use the LLM to generate a Python script that reads & summarizes a file,
        then execute that script inside the sandbox and return stdout.
        """
        file_size_mb = file_size / (1024 * 1024)

        prompt = PYTHON_READER_PROMPT.format(
            file_path=file_path,
            file_size_mb=f"{file_size_mb:.2f}",
            max_chars=max_chars,
        )

        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0
        response_text = response.content
        if active_ctx and active_logger:
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": node_name,
                "sub_call": f"read_file({os.path.basename(file_path)})",
                "prompt_len": len(prompt),
                "response_len": len(response_text),
                "elapsed_s": round(elapsed, 2),
            })
        generated_code = _extract_code(response_text, language="python")

        logger.debug(f"Generated reader code for {file_path}:\n{generated_code}")

        success, stdout, stderr = _sandbox_tool_call(
            sandbox, "sandbox_exec_python", node_name,
            code=generated_code, timeout=60,
        )

        if stdout:
            result = stdout
            if len(result) > max_chars:
                result = result[:max_chars - 3] + "..."
            logger.debug(f"File read OK: {file_path} ({len(result)} chars)")
        else:
            logger.error(f"Error reading file {file_path}: {stderr}")
            result = f"Error reading file: {stderr}"

        return result

    # ─── Node 1: scan_data ───────────────────────────────────────────────

    @node_metrics(active_ctx, active_logger, "scan_data")
    def scan_data(state: PerceptionAgentState) -> dict:
        """
        Scan the input data folder, group similar files, and use the LLM to
        read/summarize each file's content.

        Returns:
            {"data_prompt": str}
        """
        logger.info("─── [Perception Agent] scan_data ───")

        input_folder = state["input_data_folder"]
        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))
        sandbox = _get_sandbox_client(config)


        # 1. Collect all files from the sandbox (local path case)
        all_files_with_sizes = _get_all_files_sandbox(input_folder, sandbox, active_logger, active_ctx)
        all_files = [(rel, abs_path) for rel, abs_path, _ in all_files_with_sizes]
        file_sizes = {abs_path: size for _, abs_path, size in all_files_with_sizes}

        if not all_files:
            logger.error(f"  No files found in sandbox folder {input_folder}")
            return {
                "data_prompt": (
                    f"Absolute path to the folder: {input_folder}\n\n"
                    f"Unable to list files in this folder. It may be empty or inaccessible."
                ),
                "input_data_folder": input_folder
            }

        logger.info(f"  Found {len(all_files)} files in sandbox folder {input_folder}")

        # 2. Group by folder structure + extension
        file_groups = _group_similar_files(all_files)
        logger.info(f"  Grouped into {len(file_groups)} patterns")

        # 3. Read files via LLM
        file_contents = {}
        for pattern, group_files in file_groups.items():
            pattern_path = _pattern_to_path(pattern, input_folder)
            logger.info(f"  Processing pattern: {pattern_path} ({len(group_files)} files)")

            if len(group_files) > MAX_FILE_GROUP_SIZE_TO_SHOW:
                num_examples = min(NUM_EXAMPLE_FILES_TO_SHOW, len(group_files))
                example_files = random.sample(group_files, num_examples)

                group_info = (
                    f"Group pattern: {pattern_path} (total {len(group_files)} files)\n"
                    "Example files:"
                )
                example_contents = []
                for rel_path, abs_path in example_files:
                    logger.info(f"    Reading example: {abs_path}")
                    size = file_sizes.get(abs_path, 0)
                    content = _read_file_via_llm(llm, abs_path, MAX_CHARS_PER_FILE, size, sandbox, node_name="scan_data")
                    example_contents.append(f"Absolute path: {abs_path}\nContent:\n{content}")

                file_contents[group_info] = "\n-----\n".join(example_contents)
            else:
                for rel_path, abs_path in group_files:
                    file_info = f"Absolute path: {abs_path}"
                    logger.info(f"    Reading: {abs_path}")
                    size = file_sizes.get(abs_path, 0)
                    file_contents[file_info] = _read_file_via_llm(llm, abs_path, MAX_CHARS_PER_FILE, size, sandbox, node_name="scan_data")

        # 4. Assemble the data prompt
        separator = "-" * 10
        data_prompt = f"Absolute path to the folder: {input_folder}\n\nFiles structures:\n\n{separator}\n\n"
        for info, content in file_contents.items():
            data_prompt += f"{info}\nContent:\n{content}\n{separator}\n"

        logger.info(f"  data_prompt assembled: {len(data_prompt)} chars")
        return {"data_prompt": data_prompt, "input_data_folder": input_folder}

    # ─── Node 2: find_description_files ──────────────────────────────────

    @node_metrics(active_ctx, active_logger, "find_description_files")
    def find_description_files(state: PerceptionAgentState) -> dict:
        """
        Use the LLM to identify description/README files from the data prompt.

        Returns:
            {"description_files": list[str]}
        """
        logger.info("─── [Perception Agent] find_description_files ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        prompt = DESCRIPTION_FILE_RETRIEVER_PROMPT.format(data_prompt=state["data_prompt"])

        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0
        content = response.content
        if active_ctx and active_logger:
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "find_description_files",
                "prompt_len": len(prompt),
                "response_len": len(content),
                "elapsed_s": round(elapsed, 2),
            })

        # Parse: look for "Description Files:" section and extract paths
        description_files = []
        in_section = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "description files:" in stripped.lower():
                in_section = True
                continue
            if in_section and stripped:
                filename = stripped.strip("- []").strip()
                if filename:
                    description_files.append(filename)

        logger.info(f"  Found {len(description_files)} description files:")
        for f in description_files:
            logger.info(f"    → {f}")

        return {"description_files": description_files}

    # ─── Node 3: generate_task_description ───────────────────────────────

    @node_metrics(active_ctx, active_logger, "generate_task_description")
    def generate_task_description(state: PerceptionAgentState) -> dict:
        """
        Generate a concise task description from data prompt + description files.

        Returns:
            {"task_description": str}
        """
        logger.info("─── [Perception Agent] generate_task_description ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        # Read description file contents from the sandbox
        sandbox = _get_sandbox_client(config)
        file_contents = []
        for filepath in state.get("description_files", []):
            try:
                content = _sandbox_tool_call(
                    sandbox, "sandbox_read_file", "generate_task_description",
                    path=filepath,
                )
                if content is None:
                    raise RuntimeError("read_file_sync returned None")
                file_contents.append(content)
                logger.info(f"  Read {filepath} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"  Could not read {filepath}: {e}")

        description_file_contents = (
            "\n\n".join(file_contents) if file_contents
            else "No description file contents could be read."
        )

        user_input = state.get("user_input", "")

        prompt = TASK_DESCRIPTOR_PROMPT.format(
            user_input=user_input,
            data_prompt=state["data_prompt"],
            description_file_contents=description_file_contents,
        )

        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0
        response_text = response.content
        if active_ctx and active_logger:
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "generate_task_description",
                "prompt_len": len(prompt),
                "response_len": len(response_text),
                "elapsed_s": round(elapsed, 2),
            })
        task_description = response_text.strip() or "Failed to generate task description."

        logger.info(f"  Task description ({len(task_description)} chars):")
        logger.info(f"  {task_description[:300]}...")

        return {"task_description": task_description}

    # ─── Node 4: select_tools ────────────────────────────────────────────

    @node_metrics(active_ctx, active_logger, "select_tools")
    def select_tools(state: PerceptionAgentState) -> dict:
        """
        Select and rank ML tools based on task + data.

        Returns:
            {"selected_tools": list[str], "current_tool": str, "tool_prompt": str}
        """
        logger.info("─── [Perception Agent] select_tools ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        registry = _ToolRegistry(config.get("tool_registry_path"))
        tools_info = registry.format_tools_info()
        logger.debug(f"  Available tools:\n{tools_info}")

        prompt = TOOL_SELECTOR_PROMPT.format(
            task_description=state["task_description"],
            data_prompt=state["data_prompt"],
            tools_info=tools_info,
        )

        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0
        content = response.content
        if active_ctx and active_logger:
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "select_tools",
                "prompt_len": len(prompt),
                "response_len": len(content),
                "response": content,
                "elapsed_s": round(elapsed, 2),
            })

        # Parse ranked libraries
        ranked_section = re.search(r"RANKED_LIBRARIES:(.*?)$", content, re.IGNORECASE | re.DOTALL)
        prioritized_tools = []
        available_names = set(registry.list_tools())

        if ranked_section:
            items = re.findall(r"^\s*\d+\.\s*(.+?)$", ranked_section.group(1), re.MULTILINE)
            for item in items:
                name = item.strip()
                if name in available_names:
                    prioritized_tools.append(name)
                else:
                    # Closest match fallback
                    lower_to_orig = {a.lower(): a for a in available_names}
                    matches = difflib.get_close_matches(name.lower(), list(lower_to_orig.keys()), n=1, cutoff=0.4)
                    if matches:
                        closest = lower_to_orig[matches[0]]
                        logger.warning(f"  Tool '{name}' not found; using closest: '{closest}'")
                        prioritized_tools.append(closest)
                    else:
                        logger.warning(f"  Tool '{name}' not found and no close match. Skipping.")

        if not prioritized_tools:
            raise ValueError(
                "No valid ML tools were selected or parsed by the Perception Agent. "
                "Ensure that the input data path is accessible to the Perception Agent (e.g. S3 URI or mounted path)."
            )

        current_tool = prioritized_tools[0]
        tool_prompt = registry.get_tool_prompt(current_tool)

        logger.info(f"  Ranked tools: {prioritized_tools}")
        logger.info(f"  Selected tool: {current_tool}")
        logger.debug(f"  Tool prompt ({len(tool_prompt)} chars): {tool_prompt[:200]}...")

        return {
            "selected_tools": prioritized_tools,
            "current_tool": current_tool,
            "tool_prompt": tool_prompt,
        }

    # ─── Graph Assembly ──────────────────────────────────────────────────

    graph = StateGraph(PerceptionAgentState)

    graph.add_node("scan_data", scan_data)
    graph.add_node("find_description_files", find_description_files)
    graph.add_node("generate_task_description", generate_task_description)
    graph.add_node("select_tools", select_tools)

    graph.add_edge(START, "scan_data")
    graph.add_edge("scan_data", "find_description_files")
    graph.add_edge("find_description_files", "generate_task_description")
    graph.add_edge("generate_task_description", "select_tools")
    graph.add_edge("select_tools", END)

    return graph.compile()
