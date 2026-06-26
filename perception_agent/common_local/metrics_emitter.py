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
    """Emit a structured JSON event with auto-generated timestamp."""
    event["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f")
    logger.info(json.dumps(event))


def node_metrics(ctx: MetricsContext, logger: logging.Logger, node_name: str):
    """Decorator for LangGraph node functions."""
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(state, *args, **kwargs):
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
    """Decorator for agent entrypoint functions."""
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(payload, *args, **kwargs):
                ctx.init_from_payload(payload)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = await func(payload, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)

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
                ctx.init_from_payload(payload)

                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()

                result = func(payload, *args, **kwargs)

                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)

                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)

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
