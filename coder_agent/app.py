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
logger = logging.getLogger("coder_agent.app")

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    metric_logger.addHandler(_h)

from common_local.metrics_context import MetricsContext
from common_local.metrics_emitter import emit_event
from common_local.logging_callback import SessionMetricsCallback

ctx = MetricsContext(agent_id="coder_agent")

from coder_agent.agent import build_coder_agent_graph, CoderAgentState
from coder_agent.sandbox_client import SandboxClient

# Pre-compile the LangGraph for code generation (cold start)
_graph = build_coder_agent_graph(ctx=ctx, metric_logger=metric_logger)
logger.info("Coder Agent synchronous LangGraph compiled successfully.")

app = BedrockAgentCoreApp()

async def _run_coder_core(payload: Dict[str, Any], invocation_start_ms: int) -> Dict[str, Any]:
    """Runs LLM generation, writes files, executes synchronously inside sandbox, and evaluates."""
    # Build initial state from payload
    config = {
        "llm": {
            "model": os.environ.get("LLM_MODEL", "gpt-4o"),
            "temperature": 0.1,
        },
        "mcts": {
            "continuous_improvement": True,
        },
        "tool_registry_path": str(_project_root / "MLorchestrator" / "shared" / "tools_registry" if (_project_root / "MLorchestrator" / "shared" / "tools_registry").exists() else _project_root / "tools_registry")
    }

    incoming_config = payload.get("config", {})
    if isinstance(incoming_config, dict):
        for k, v in incoming_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k] = {**config[k], **v}
            else:
                config[k] = v

    sandbox_url = config.get("sandbox_url") or os.environ.get("SANDBOX_URL")
    sandbox = SandboxClient(sandbox_url)

    initial_state = {
        "config": config,
        "task_description": payload.get("task_description", ""),
        "data_prompt": payload.get("data_prompt", ""),
        "user_input": payload.get("user_input", ""),
        "current_tool": payload.get("current_tool", "machine learning"),
        "tool_prompt": payload.get("tool_prompt", ""),
        "tutorial_prompt": payload.get("tutorial_prompt", ""),
        "all_error_analyses": payload.get("all_error_analyses", []),
        "previous_python_code": payload.get("previous_python_code") or payload.get("parent_code", ""),
        "previous_bash_script": payload.get("previous_bash_script") or payload.get("parent_bash", ""),
        "stage": payload.get("stage", "evolve"),
        "iteration": payload.get("iteration", 0),
        "node_id": payload.get("node_id"),
        "sandbox_client": sandbox
    }

    # Execute LangGraph generation & execution workflow
    langgraph_config = {
        "callbacks": [SessionMetricsCallback(ctx=ctx, metric_logger=metric_logger)]
    }
    
    t0 = time.time()
    result = await _graph.ainvoke(initial_state, config=langgraph_config)
    elapsed = time.time() - t0
    
    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type": "psutil_metrics_graph",
        "graph_name": "coder_agent_run",
        "graph_e2e_s": elapsed,
        "step_count": 3,
        "iteration_count": payload.get("iteration", 0)
    })

    return result


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    Main AgentCore app entrypoint.
    Executes code generation, blocking execution inside sandbox, and evaluates results.
    """
    invocation_start_ms = int(time.time() * 1000)
    ctx.init_from_payload(payload)
    
    logger.info("Coder Agent Invoked synchronously")

    try:
        result = asyncio.run(_run_coder_core(payload, invocation_start_ms))
        
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation",
            "status": "COMPLETED",
            "invocation_start_ms": invocation_start_ms,
            "total_ms": int(time.time() * 1000) - invocation_start_ms,
        })

        return {
            "status": "COMPLETED",
            "python_code": result.get("python_code", ""),
            "bash_script": result.get("bash_script", ""),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "decision": result.get("decision", "FIX"),
            "validation_score": result.get("validation_score"),
            "error_message": result.get("error_message", ""),
            "error_analysis": result.get("error_analysis", "")
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
