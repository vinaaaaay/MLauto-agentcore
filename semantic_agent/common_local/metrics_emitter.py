"""
Standardized metrics emission via decorators.

Two decorators:
    @node_metrics   — wraps LangGraph node functions
    @graph_metrics  — wraps agent entrypoint functions

One utility:
    emit_event()    — structured JSON emission with auto-timestamp

Canonical event types emitted:
    psutil_metrics_node   — timing + RAM per node execution
    psutil_metrics_graph  — timing + RAM + step_count per graph execution

Other canonical events (llm_call, mcp_tool_execution) are emitted by
logging_callback.py and mcp_tool_factory.py respectively.

No agent-specific knowledge in this file.
"""

import time
import json
import psutil
import functools
import logging
import asyncio
import inspect
from datetime import datetime, timezone
from typing import Callable

from .metrics_context import MetricsContext


def emit_event(logger: logging.Logger, event: dict):
    """
    Emit a structured JSON event with auto-generated timestamp.

    Use for canonical metric events and debug/observability events.
    Every event goes to stdout as a single JSON line.
    """
    event["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f")
    logger.info(json.dumps(event))


def node_metrics(ctx: MetricsContext, logger: logging.Logger, node_name: str):
    """
    Decorator for LangGraph node functions.

    Handles:
        - Setting current node name in context (so llm_call and
          mcp_tool_execution events inherit the correct node_name)
        - Measuring wall-clock time
        - Measuring peak RAM (approximated via RSS before/after)
        - Emitting psutil_metrics_node event

    The decorated function contains ONLY business logic.

    Usage:
        @node_metrics(ctx, metric_logger, "planner")
        def planner_node(state):
            # pure business logic
            response = model.invoke(messages)
            return {"messages": [response], ...}
    """
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(state, *args, **kwargs):
                # Set node name so callbacks (llm_call, mcp) pick it up
                ctx.node_name.set(node_name)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = await func(state, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":  "psutil_metrics_node",
                    "node_name":   node_name,
                    "node_e2e_s":  round(elapsed, 4),
                    "peak_RAM_GB": round(peak_mem / (1024**3), 4),
                })

                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(state, *args, **kwargs):
                # Set node name so callbacks (llm_call, mcp) pick it up
                ctx.node_name.set(node_name)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = func(state, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":  "psutil_metrics_node",
                    "node_name":   node_name,
                    "node_e2e_s":  round(elapsed, 4),
                    "peak_RAM_GB": round(peak_mem / (1024**3), 4),
                })

                return result
            return sync_wrapper
    return decorator


def graph_metrics(ctx: MetricsContext, logger: logging.Logger, graph_name: str):
    """
    Decorator for agent entrypoint functions.

    Handles:
        - Initializing tracing context from incoming payload
        - Measuring wall-clock time for entire graph execution
        - Measuring peak RAM
        - Extracting step_count and iteration_count from result
        - Emitting psutil_metrics_graph event

    Must be applied INSIDE @app.entrypoint (closer to the function):

        @app.entrypoint
        @graph_metrics(ctx, metric_logger, "planner")
        def handle(payload):
            ...
    """
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(payload, *args, **kwargs):
                # Initialize all tracing context from the incoming payload
                ctx.init_from_payload(payload)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = await func(payload, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                # Extract step_count from result (check common locations)
                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)
                    agent_state = result.get("agent_state", {})
                    if isinstance(agent_state, dict):
                        step_count = step_count or agent_state.get("step_count", 0)
                        iteration_count = iteration_count or agent_state.get("iteration_count", 0) or agent_state.get("round_count", 0)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":      "psutil_metrics_graph",
                    "graph_name":      graph_name,
                    "graph_e2e_s":     round(elapsed, 4),
                    "peak_RAM_GB":     round(peak_mem / (1024**3), 4),
                    "step_count":      step_count,
                    "iteration_count": iteration_count,
                })

                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(payload, *args, **kwargs):
                # Initialize all tracing context from the incoming payload
                ctx.init_from_payload(payload)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = func(payload, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                # Extract step_count from result (check common locations)
                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)
                    agent_state = result.get("agent_state", {})
                    if isinstance(agent_state, dict):
                        step_count = step_count or agent_state.get("step_count", 0)
                        iteration_count = iteration_count or agent_state.get("iteration_count", 0) or agent_state.get("round_count", 0)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":      "psutil_metrics_graph",
                    "graph_name":      graph_name,
                    "graph_e2e_s":     round(elapsed, 4),
                    "peak_RAM_GB":     round(peak_mem / (1024**3), 4),
                    "step_count":      step_count,
                    "iteration_count": iteration_count,
                })

                return result
            return sync_wrapper
    return decorator