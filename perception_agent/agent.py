"""
Single build agent compiling the Perception Agent StateGraph with inline nodes.

Graph flow:
  START → scan_data → find_description_files → generate_task_description
        → select_tools → END
"""

import logging
import os
import random
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from .state import PerceptionAgentState
from .prompts import (
    PYTHON_READER_PROMPT,
    DESCRIPTION_FILE_RETRIEVER_PROMPT,
    TASK_DESCRIPTOR_PROMPT,
    TOOL_SELECTOR_PROMPT,
)
from .utils import (
    _get_llm,
    _get_all_files,
    _get_sandbox_client,
    _get_all_files_sandbox,
    _execute_code_sandbox,
    _group_similar_files,
    _pattern_to_path,
    _extract_code,
    _execute_code,
    _ToolRegistry,
    MAX_CHARS_PER_FILE,
    MAX_FILE_GROUP_SIZE_TO_SHOW,
    NUM_EXAMPLE_FILES_TO_SHOW,
    DEFAULT_LIBRARY,
)

import json
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from common.metrics_context import MetricsContext
from common.metrics_emitter import node_metrics, emit_event

metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="perception_agent")


def build_perception_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Perception Agent LangGraph.

    Contains all graph nodes inline to encapsulate execution scope,
    matching the single build agent design pattern used across FAME.
    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")

    # ─── Helper ──────────────────────────────────────────────────────────

    def _sandbox_tool_call(sandbox, tool_name: str, node_name: str, **kwargs):
        """
        Instrumented wrapper around sandbox operations.
        Emits tool_call events matching SessionMetricsCallback format.
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

    # ─── Node 1: scan_data ───────────────────────────────────────────────

    @node_metrics(active_ctx, active_logger, "scan_data")
    def scan_data(state: PerceptionAgentState) -> dict:
        """
        Scan directory, group similar files, and read files via LLM.

        Maps to: DataPerceptionAgent.__call__()

        Returns:
            {"data_prompt": str}
        """
        logger.info("─── [Perception Agent] scan_data ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        # Setup sandbox
        sandbox = _get_sandbox_client(config)

        input_folder = state["input_data_folder"]
        output_folder = state.get("output_folder", "/tmp/perception_output")

        from .utils import _LLMCallLogger
        llm_logger = _LLMCallLogger(output_folder, active_ctx, active_logger)

        logger.info(f"  Scanning input data folder in sandbox: {input_folder}")
        files = _get_all_files_sandbox(input_folder, sandbox)
        logger.info(f"  Found {len(files)} files in sandbox.")

        if not files:
            return {"data_prompt": "No files found in the input folder."}

        # Convert sizes to MB and drop size for grouping
        files_for_grouping = [(rel_path, abs_path) for rel_path, abs_path, _ in files]
        groups = _group_similar_files(files_for_grouping)

        # Build data summary prompt section
        data_summary_lines = []
        for pattern, group_files in groups.items():
            pattern_str = _pattern_to_path(pattern, input_folder)
            data_summary_lines.append(f"Pattern: {pattern_str} ({len(group_files)} files)")

            # Sample files to show
            sampled = random.sample(group_files, min(len(group_files), NUM_EXAMPLE_FILES_TO_SHOW))
            for rel_path, abs_path in sampled:
                # Find size from original files list
                size_bytes = next(sz for r, _, sz in files if r == rel_path)
                size_mb = round(size_bytes / (1024 * 1024), 2)

                data_summary_lines.append(f"  Example: {abs_path} ({size_mb} MB)")

                # Generate code to read file in sandbox
                prompt = PYTHON_READER_PROMPT.format(
                    file_path=abs_path,
                    file_size_mb=size_mb,
                    max_chars=MAX_CHARS_PER_FILE,
                )

                llm_response = llm_logger.call(llm, prompt, node_name="scan_data")
                python_code = _extract_code(llm_response, language="python")

                # Run reader script inside sandbox
                logger.info(f"  Executing reader code in sandbox for {abs_path}")
                success, stdout, stderr = _sandbox_tool_call(
                    sandbox, "sandbox_exec_shell", "scan_data",
                    command=python_code,
                )

                if success:
                    content = stdout.strip() or "Empty output from reader."
                    # Escape braces to prevent string formatting issues later
                    content_escaped = content.replace("{", "{{").replace("}", "}}")
                    data_summary_lines.append(f"  Content:\n{content_escaped}")
                else:
                    logger.warning(f"  Reader failed for {abs_path}: {stderr}")
                    data_summary_lines.append(f"  Reader failed: {stderr}")

            data_summary_lines.append("")

        data_prompt = "\n".join(data_summary_lines)
        logger.info(f"  Data perception prompt generated ({len(data_prompt)} chars).")

        return {"data_prompt": data_prompt}

    # ─── Node 2: find_description_files ──────────────────────────────────

    @node_metrics(active_ctx, active_logger, "find_description_files")
    def find_description_files(state: PerceptionAgentState) -> dict:
        """
        Identify project description/readme files using LLM.

        Maps to: DescriptionFileRetrieverAgent.__call__()

        Returns:
            {"description_files": list[str]}
        """
        logger.info("─── [Perception Agent] find_description_files ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        prompt = DESCRIPTION_FILE_RETRIEVER_PROMPT.format(
            data_prompt=state["data_prompt"]
        )

        t0 = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - t0
        response_text = response.content
        if active_ctx and active_logger:
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "find_description_files",
                "prompt_len": len(prompt),
                "response_len": len(response_text),
                "elapsed_s": round(elapsed, 2),
            })

        # Parse description files
        desc_files = []
        for line in response_text.split("\n"):
            if line.strip().startswith("Description Files:"):
                # Handle single line list if formatted as JSON
                files_str = line.split("Description Files:")[1].strip()
                if files_str.startswith("[") and files_str.endswith("]"):
                    try:
                        desc_files.extend(json.loads(files_str))
                    except json.JSONDecodeError:
                        pass
            elif line.strip() and not line.strip().startswith("-") and "/" in line:
                desc_files.append(line.strip())

        # Clean duplicates
        desc_files = list(set(desc_files))
        logger.info(f"  Identified {len(desc_files)} description files: {desc_files}")

        return {"description_files": desc_files}

    # ─── Node 3: generate_task_description ────────────────────────────────

    @node_metrics(active_ctx, active_logger, "generate_task_description")
    def generate_task_description(state: PerceptionAgentState) -> dict:
        """
        Generate a concise DS task description from data + description files.

        Maps to: TaskDescriptorAgent.__call__()

        Returns:
            {"task_description": str}
        """
        logger.info("─── [Perception Agent] generate_task_description ───")

        config = state.get("config", {})
        llm = _get_llm(config.get("llm"))

        # Setup sandbox to read files
        sandbox = _get_sandbox_client(config)

        # Read description file contents from sandbox
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

        Maps to: ToolSelectorAgent.__call__()

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
                    closest = min(available_names, key=lambda x: len(set(x.lower()) ^ set(name.lower())))
                    logger.warning(f"  Tool '{name}' not found; using closest: '{closest}'")
                    prioritized_tools.append(closest)

        if not prioritized_tools:
            logger.warning(f"  Could not parse tools from LLM response. Defaulting to '{DEFAULT_LIBRARY}'.")
            prioritized_tools = [DEFAULT_LIBRARY]

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
