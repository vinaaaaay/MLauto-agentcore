import time
import json
import psutil
import functools
import logging
import inspect
from datetime import datetime, timezone
from typing import Callable

from .metrics_context import MetricsContext


def emit_event(logger: logging.Logger, event: dict):
    """Emit a structured JSON event with auto-generated timestamp."""
    event["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f")
    logger.info(json.dumps(event))


def _get_process_metrics(process: psutil.Process):
    """Helper to get memory, cpu, and io safely."""
    mem = process.memory_info().rss
    cpu = process.cpu_times()
    try:
        io = process.io_counters()
    except Exception:
        io = None
    return mem, cpu, io


def _compute_metrics(process: psutil.Process, t0: float, start_mem: int, start_cpu, start_io):
    elapsed = time.time() - t0
    peak_mem = max(start_mem, process.memory_info().rss)
    end_cpu = process.cpu_times()
    
    active_cpu_s = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    wait_time_s = max(0, elapsed - active_cpu_s)
    
    io_read_mb = 0.0
    io_write_mb = 0.0
    if start_io:
        try:
            end_io = process.io_counters()
            io_read_mb = (end_io.read_bytes - start_io.read_bytes) / (1024 * 1024)
            io_write_mb = (end_io.write_bytes - start_io.write_bytes) / (1024 * 1024)
        except Exception:
            pass

    return {
        "e2e_s": round(elapsed, 4),
        "active_cpu_s": round(active_cpu_s, 4),
        "wait_time_s": round(wait_time_s, 4),
        "io_read_MB": round(io_read_mb, 4),
        "io_write_MB": round(io_write_mb, 4),
        "peak_RAM_GB": round(peak_mem / (1024**3), 4)
    }


def node_metrics(ctx: MetricsContext, logger: logging.Logger, node_name: str):
    """Decorator for LangGraph node functions."""
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(state, *args, **kwargs):
                ctx.node_name.set(node_name)
                process = psutil.Process()
                start_mem, start_cpu, start_io = _get_process_metrics(process)
                t0 = time.time()

                result = await func(state, *args, **kwargs)

                metrics = _compute_metrics(process, t0, start_mem, start_cpu, start_io)
                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":  "psutil_metrics_node",
                    "node_name":   node_name,
                    "node_e2e_s":  metrics["e2e_s"],
                    "active_cpu_s": metrics["active_cpu_s"],
                    "wait_time_s": metrics["wait_time_s"],
                    "io_read_MB": metrics["io_read_MB"],
                    "io_write_MB": metrics["io_write_MB"],
                    "peak_RAM_GB": metrics["peak_RAM_GB"],
                })
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(state, *args, **kwargs):
                ctx.node_name.set(node_name)
                process = psutil.Process()
                start_mem, start_cpu, start_io = _get_process_metrics(process)
                t0 = time.time()

                result = func(state, *args, **kwargs)

                metrics = _compute_metrics(process, t0, start_mem, start_cpu, start_io)
                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":  "psutil_metrics_node",
                    "node_name":   node_name,
                    "node_e2e_s":  metrics["e2e_s"],
                    "active_cpu_s": metrics["active_cpu_s"],
                    "wait_time_s": metrics["wait_time_s"],
                    "io_read_MB": metrics["io_read_MB"],
                    "io_write_MB": metrics["io_write_MB"],
                    "peak_RAM_GB": metrics["peak_RAM_GB"],
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
                start_mem, start_cpu, start_io = _get_process_metrics(process)
                t0 = time.time()

                result = await func(payload, *args, **kwargs)

                metrics = _compute_metrics(process, t0, start_mem, start_cpu, start_io)
                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":      "psutil_metrics_graph",
                    "graph_name":      graph_name,
                    "graph_e2e_s":     metrics["e2e_s"],
                    "active_cpu_s":    metrics["active_cpu_s"],
                    "wait_time_s":     metrics["wait_time_s"],
                    "io_read_MB":      metrics["io_read_MB"],
                    "io_write_MB":     metrics["io_write_MB"],
                    "peak_RAM_GB":     metrics["peak_RAM_GB"],
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
                start_mem, start_cpu, start_io = _get_process_metrics(process)
                t0 = time.time()

                result = func(payload, *args, **kwargs)

                metrics = _compute_metrics(process, t0, start_mem, start_cpu, start_io)
                step_count = 0
                iteration_count = 0
                if isinstance(result, dict):
                    step_count = step_count or result.get("step_count", 0)
                    iteration_count = iteration_count or result.get("iteration_count", 0) or result.get("round_count", 0)

                emit_event(logger, {
                    **ctx.snapshot(),
                    "event_type":      "psutil_metrics_graph",
                    "graph_name":      graph_name,
                    "graph_e2e_s":     metrics["e2e_s"],
                    "active_cpu_s":    metrics["active_cpu_s"],
                    "wait_time_s":     metrics["wait_time_s"],
                    "io_read_MB":      metrics["io_read_MB"],
                    "io_write_MB":     metrics["io_write_MB"],
                    "peak_RAM_GB":     metrics["peak_RAM_GB"],
                    "step_count":      step_count,
                    "iteration_count": iteration_count,
                })
                return result
            return sync_wrapper
    return decorator
