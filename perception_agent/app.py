import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# ─── Load Environment ───
_curr_dir = Path(__file__).resolve().parent
load_dotenv(_curr_dir / ".env")

# Ensure package root is in sys.path
_project_root = _curr_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ─── Logging Setup ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perception_agent.app")

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    metric_logger.addHandler(_h)

from common_local.metrics_context import MetricsContext
from common_local.metrics_emitter import emit_event
from common_local.logging_callback import SessionMetricsCallback

ctx = MetricsContext(agent_id="perception_agent")

from perception_agent.agent import build_perception_agent_graph

# Pre-compile the LangGraph (cold start)
_graph = build_perception_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("Perception Agent LangGraph compiled successfully.")

app = BedrockAgentCoreApp()


async def _run_perception_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runs all 4 perception nodes synchronously inside LangGraph."""
    config = {
        "llm": {
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "temperature": 0.1,
        },
    }

    incoming_config = payload.get("config", {})
    if isinstance(incoming_config, dict):
        for k, v in incoming_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k] = {**config[k], **v}
            else:
                config[k] = v

    initial_state = {
        "config": config,
        "input_data_folder": payload.get("input_data_folder", ""),
        "output_folder": payload.get("output_folder", ""),
        "user_input": payload.get("user_input", ""),
        "all_error_analyses": payload.get("all_error_analyses", []),
    }

    langgraph_config = {
        "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
    }

    import resource
    t0 = time.time()
    result = await _graph.ainvoke(initial_state, config=langgraph_config)
    elapsed = time.time() - t0

    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type": "resource_metrics",
        "graph_name": "perception_agent_run",
        "graph_e2e_s": round(elapsed, 4),
        "peak_ram_MB": round(peak_ram_mb, 4),
        "step_count": 4,
    })
    logger.info(f"[Perception Agent Run] E2E Time: {elapsed:.2f}s | Peak RAM: {peak_ram_mb:.2f} MB")

    return result


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    Main AgentCore app entrypoint.
    Runs the full perception pipeline and returns structured output.
    """
    invocation_start_ms = int(time.time() * 1000)
    ctx.init_from_payload(payload)

    logger.info("Perception Agent invoked.")

    try:
        result = asyncio.run(_run_perception_core(payload))

        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation",
            "status": "COMPLETED",
            "invocation_start_ms": invocation_start_ms,
            "total_ms": int(time.time() * 1000) - invocation_start_ms,
        })

        return {
            "status": "COMPLETED",
            "data_prompt": result.get("data_prompt", ""),
            "description_files": result.get("description_files", []),
            "task_description": result.get("task_description", ""),
            "selected_tools": result.get("selected_tools", []),
            "current_tool": result.get("current_tool", ""),
            "tool_prompt": result.get("tool_prompt", ""),
            "input_data_folder": result.get("input_data_folder", ""),
        }

    except Exception as exc:
        logger.error(f"[handle] Execution error: {exc}", exc_info=True)
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "invocation_start_ms": invocation_start_ms,
            "total_ms": int(time.time() * 1000) - invocation_start_ms,
        })
        return {
            "status": "FAILED",
            "error": str(exc),
        }


if __name__ == "__main__":
    app.run()
