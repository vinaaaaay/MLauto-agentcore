"""
Semantic Agent — Bedrock AgentCore Entrypoint.

Mirrors the actor pattern from FAME-fork/reflexion-app1/functions/actor/lambda_handler.py:
  - _run_semantic_core()  async core logic (graph invocation, state building, metrics)
  - handle()              @app.entrypoint — wraps core with tracing init, error handling

This is the AgentCore deployment path.

Payload schema (JSON):
    {
        "task_description":   str,        # required
        "current_tool":       str,        # required — ML library to search for
        "all_error_analyses": list[str],  # optional
        "data_prompt":        str,        # optional
        "user_input":         str,        # optional
        "session_id":         str,        # optional — for tracing propagation
        "config": {                       # optional — overrides agent defaults
            "llm": {"model": str, "temperature": float},
            "tutorials": {
                "num_tutorial_retrievals": int,
                "condense_tutorials":      bool,
                "max_num_tutorials":       int
            }
        },
        "tracing": {                      # optional — injected by parent agent
            "context_id":    str,
            "parent_span_id": str
        }
    }

Response schema (JSON):
    {
        "tutorial_prompt": str,   # formatted tutorial content for the coder agent
        "status":          str    # "COMPLETED" or "FAILED"
    }
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# ─── Environment ──────────────────────────────────────────────────────────────
_curr_dir = Path(__file__).resolve().parent
load_dotenv(_curr_dir / ".env")

# ─── Ensure package root is importable ────────────────────────────────────────
_project_root = _curr_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─── AgentCore runtime ────────────────────────────────────────────────────────
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("semantic_agent.app")

# Structured metrics logger — one JSON line per event, consumed by AgentCore observability
metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    metric_logger.addHandler(_h)

# ─── Metrics infrastructure ────────────────────────────────────────────────────
from common_local.metrics_context import MetricsContext
from common_local.metrics_emitter import emit_event
from common_local.logging_callback import SessionMetricsCallback

ctx = MetricsContext(agent_id="semantic_agent")

# ─── Agent graph (compiled once at cold start) ────────────────────────────────
from semantic_agent.agent import build_semantic_agent_graph

_graph = build_semantic_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("[semantic_agent.app] LangGraph compiled successfully (cold start).")

# ─── Config ───────────────────────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
VECTOR_STORE_ARN = os.environ.get("VECTOR_STORE_URL", "")


# ══════════════════════════════════════════════════════════════════════════════
#  Core async logic  (mirrors _run_actor_core in the actor pattern)
# ══════════════════════════════════════════════════════════════════════════════

async def _run_semantic_core(
    payload: Dict[str, Any],
    invocation_start_ms: int,
) -> Dict[str, Any]:
    """
    Async core for the Semantic Agent.

    Builds the LangGraph initial state from the incoming payload, invokes the
    graph, and returns the structured result.  Metrics events are emitted to
    metric_logger throughout execution.

    Parameters
    ----------
    payload            Raw dict received by the @app.entrypoint.
    invocation_start_ms  Epoch-ms timestamp captured at the very start of handle().
    """
    task_description   = payload.get("task_description", "")
    current_tool       = payload.get("current_tool", "")
    all_error_analyses = payload.get("all_error_analyses", [])
    data_prompt        = payload.get("data_prompt", "")
    user_input         = payload.get("user_input", "")
    session_id         = payload.get("session_id", "unknown")

    logger.info(
        f"[_run_semantic_core] session_id={session_id!r} "
        f"task_description={task_description[:80]!r} "
        f"current_tool={current_tool!r}"
    )

    # ── Build config ──────────────────────────────────────────────────────────
    # Agent-owned defaults.  VECTOR_STORE_ARN always comes from the environment
    # variable — callers cannot override the mcpserver endpoint.
    config: Dict[str, Any] = {
        "llm": {
            "model":       os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "temperature": 0.1,
        },
        "mcp_servers": {
            "vector_store_url": VECTOR_STORE_ARN,
        },
        "tutorials": {
            "num_tutorial_retrievals": 3,
            "condense_tutorials":      False,
            "max_num_tutorials":       2,
        },
    }
    # Deep-merge caller-supplied overrides on top of defaults
    incoming_config = payload.get("config", {})
    if isinstance(incoming_config, dict):
        for section, values in incoming_config.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section] = {**config[section], **values}
            else:
                config[section] = values
        if incoming_config:
            logger.info(f"[_run_semantic_core] Merged config sections: {list(incoming_config.keys())}")

    # Enforce env-sourced ARN regardless of what caller sent
    config["mcp_servers"]["vector_store_url"] = VECTOR_STORE_ARN

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: Dict[str, Any] = {
        "config":             config,
        "output_folder":      "/tmp/agentcore_output",
        "task_description":   task_description,
        "current_tool":       current_tool,
        "all_error_analyses": all_error_analyses,
        "data_prompt":        data_prompt,
        "user_input":         user_input,
    }
    # Pass through any extra fields the caller injected
    _known_keys = {
        "task_description", "current_tool", "all_error_analyses", "data_prompt",
        "user_input", "config", "tracing", "session_id", "query_id",
    }
    for k, v in payload.items():
        if k not in _known_keys:
            initial_state[k] = v

    # ── Invoke graph ──────────────────────────────────────────────────────────
    langgraph_config = {
        "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
    }

    import psutil
    process = psutil.Process()
    start_cpu = process.cpu_times()
    try:
        start_io = process.io_counters()
    except Exception:
        start_io = None
        
    t0 = time.time()
    result = await _graph.ainvoke(initial_state, config=langgraph_config)
    elapsed_s = round(time.time() - t0, 4)

    tutorial_prompt = result.get("tutorial_prompt", "")

    logger.info(
        f"[_run_semantic_core] Done in {elapsed_s}s. "
        f"tutorial_prompt={len(tutorial_prompt)} chars"
    )
    end_cpu = psutil.Process().cpu_times()
    active_cpu_s = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    wait_time_s = max(0, elapsed_s - active_cpu_s)

    io_read_mb = 0.0
    io_write_mb = 0.0
    if start_io:
        try:
            end_io = psutil.Process().io_counters()
            io_read_mb = (end_io.read_bytes - start_io.read_bytes) / (1024 * 1024)
            io_write_mb = (end_io.write_bytes - start_io.write_bytes) / (1024 * 1024)
        except Exception:
            pass
            
    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type":  "psutil_metrics_graph",
        "graph_name":  "semantic_agent",
        "graph_e2e_s": elapsed_s,
        "active_cpu_s": round(active_cpu_s, 4),
        "wait_time_s": round(wait_time_s, 4),
        "io_read_MB": round(io_read_mb, 4),
        "io_write_MB": round(io_write_mb, 4),
        "step_count":  3,
    })

    return {
        "tutorial_prompt": tutorial_prompt,
        "session_id":      session_id,
        "current_tool":    current_tool,
        "mcp_calls":       result.get("mcp_calls", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AgentCore App  
# ══════════════════════════════════════════════════════════════════════════════

app = BedrockAgentCoreApp()


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    AgentCore entrypoint — called by the runtime harness on every invocation.

    Initializes the MetricsContext from tracing propagation in the payload,
    runs the async core, emits an invocation event, and returns the result.
    Errors are caught and returned as a FAILED status response so that the
    caller (orchestrator) can handle them gracefully.
    """
    invocation_start_ms = int(time.time() * 1000)

    # ── Initialize tracing context from incoming payload ──────────────────────
    ctx.init_from_payload(payload)
    session_id = payload.get("session_id", "unknown")

    logger.info(
        f"[handle] invocation start | session_id={session_id!r} "
        f"span_id={ctx.span_id.get()!r}"
    )

    # ── Run core (asyncio.run is safe — AgentCore runs each request in a thread)
    try:
        result = asyncio.run(_run_semantic_core(payload, invocation_start_ms))

        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type":           "invocation",
            "status":               "COMPLETED",
            "invocation_start_ms":  invocation_start_ms,
            "total_ms":             int(time.time() * 1000) - invocation_start_ms,
        })

        return {
            "tutorial_prompt": result.get("tutorial_prompt", ""),
            "status":          "COMPLETED",
            "mcp_calls":       result.get("mcp_calls", []),
        }

    except Exception as exc:
        logger.error(f"[handle] Execution error: {exc}", exc_info=True)

        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type":           "invocation",
            "status":               "FAILED",
            "error_type":           type(exc).__name__,
            "error_message":        str(exc),
            "invocation_start_ms":  invocation_start_ms,
            "total_ms":             int(time.time() * 1000) - invocation_start_ms,
        })

        return {
            "tutorial_prompt": "",
            "status":          "FAILED",
            "error":           str(exc),
            "mcp_calls":       [],
        }


if __name__ == "__main__":
    app.run()
