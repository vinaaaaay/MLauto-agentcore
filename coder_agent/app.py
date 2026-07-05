import asyncio
import logging
import os
import sys
import threading
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

# ─── In-memory result stores ─────────────────────────────────────────────────
# Background polling thread writes here; check_status reads and pops.
_completed_jobs: Dict[str, Dict[str, Any]] = {}
_failed_jobs: Dict[str, Dict[str, Any]] = {}
_job_store_lock = threading.Lock()

async def _run_coder_core(payload: Dict[str, Any], invocation_start_ms: int) -> Dict[str, Any]:
    """Runs LLM generation, writes files, executes synchronously inside sandbox via MCP, and evaluates."""
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

    current_tool = payload.get("current_tool")
    if not current_tool:
        raise ValueError("Missing 'current_tool' in payload for CodingAgent.")

    initial_state = {
        "config": config,
        "task_description": payload.get("task_description", ""),
        "data_prompt": payload.get("data_prompt", ""),
        "user_input": payload.get("user_input", ""),
        "current_tool": current_tool,
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

    import resource
    t0 = time.time()
    try:
        result = await _graph.ainvoke(initial_state, config=langgraph_config)
    finally:
        # Always close the SSE connection regardless of success or failure.
        await sandbox.close()

    elapsed = time.time() - t0
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type": "resource_metrics",
        "graph_name": "coder_agent_run",
        "graph_e2e_s": round(elapsed, 4),
        "peak_ram_MB": round(peak_ram_mb, 4),
        "step_count": 3,
        "iteration_count": payload.get("iteration", 0)
    })
    logger.info(f"[Coder Agent Run] E2E Time: {elapsed:.2f}s | Peak RAM: {peak_ram_mb:.2f} MB")

    return result


def _background_poll_and_evaluate(
    job_id: str,
    task_id: int,
    payload: Dict[str, Any],
    invocation_start_ms: int,
) -> None:
    """
    Background thread:
    1. Runs the full Coder Agent LangGraph pipeline (_run_coder_core).
       The pipeline generates code, writes it to the sandbox via MCP, then calls
       exec_sandbox (delivery=poll) which blocks the thread over the SSE connection
       until the training script finishes — no manual file polling needed.
    2. Evaluates results inline inside execute_and_evaluate node.
    3. Stores the formatted result in _completed_jobs / _failed_jobs.
    4. Marks the Bedrock async task complete so the session reverts to Healthy.
    """
    logger.info(f"[BG] Background thread started for job_id={job_id}")
    try:
        # Run the graph to completion (blocks while training runs via MCP task poll)
        result = asyncio.run(_run_coder_core(payload, invocation_start_ms))
        logger.info(f"[BG] Graph completed for job_id={job_id}. Storing result.")

        formatted = {
            "status": "COMPLETED",
            "python_code": result.get("python_code", ""),
            "bash_script": result.get("bash_script", ""),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "decision": result.get("decision", "FIX"),
            "validation_score": result.get("validation_score"),
            "error_message": result.get("error_message", ""),
            "error_analysis": result.get("error_analysis", ""),
            "error_summary": result.get("error_summary", ""),
        }
        with _job_store_lock:
            _completed_jobs[job_id] = formatted

    except Exception as exc:
        logger.error(
            f"[BG] Fatal error in background thread for job_id={job_id}: {exc}",
            exc_info=True,
        )
        with _job_store_lock:
            _failed_jobs[job_id] = {"status": "FAILED", "error": str(exc)}
    finally:
        app.complete_async_task(task_id)
        logger.info(f"[BG] Background thread finished for job_id={job_id}. Session released.")


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    Main AgentCore app entrypoint.

    Actions:
    - generate_and_run: Generate code, write to sandbox via MCP write_file, execute
      via exec_sandbox (delivery=poll, blocking over SSE), evaluate inline, register
      async task (HealthyBusy), spawn background thread, return ACCEPTED immediately.
    - check_status: In-memory lookup of the result stored by the background
      thread. Returns RUNNING until the thread completes, then COMPLETED/FAILED.
    """
    invocation_start_ms = int(time.time() * 1000)
    ctx.init_from_payload(payload)

    action = payload.get("action", "generate_and_run")
    logger.info(f"Coder Agent Invoked: action={action}")

    try:
        if action == "generate_and_run":
            # ── Predict job_id from payload (deterministic mapping) ─────────
            node_id = payload.get("node_id")
            iteration = payload.get("iteration", 0)
            job_id = f"node_{node_id}" if node_id is not None else f"iteration_{iteration}"

            # ── Phase 1: Register async task → session stays HealthyBusy ────
            # The SDK sets /ping to HealthyBusy while _active_tasks is non-empty,
            # keeping this session alive past the 15-minute idle timeout.
            task_id = app.add_async_task(
                "training_job", metadata={"job_id": job_id}
            )
            logger.info(
                f"Registered async task (id={task_id}) for job_id={job_id}. "
                f"Session will report HealthyBusy until training completes."
            )

            # ── Phase 2: Spawn background thread to perform generation + launch + poll ──
            # The thread does the LLM generation, sandbox submission, polling, and evaluation.
            bg_thread = threading.Thread(
                target=_background_poll_and_evaluate,
                args=(job_id, task_id, payload, invocation_start_ms),
                daemon=True,
                name=f"poll-{job_id}",
            )
            bg_thread.start()
            logger.info(f"Background execution thread started for job_id={job_id}.")

            emit_event(metric_logger, {
                **ctx.snapshot(),
                "event_type": "invocation",
                "status": "ACCEPTED",
                "invocation_start_ms": invocation_start_ms,
                "total_ms": int(time.time() * 1000) - invocation_start_ms,
            })

            return {
                "status": "ACCEPTED",
                "job_id": job_id,
                "python_code": "",
                "bash_script": "",
            }

        elif action == "check_status":
            # ── In-memory lookup — no sandbox calls, no cold start ───────────
            # The background thread writes to _completed_jobs / _failed_jobs
            # when training finishes. Until then we return RUNNING.
            job_id = payload.get("job_id", "")
            if not job_id:
                raise ValueError("Missing 'job_id' for check_status action.")

            with _job_store_lock:
                if job_id in _completed_jobs:
                    logger.info(f"check_status: job_id={job_id} → COMPLETED (in-memory)")
                    return _completed_jobs.pop(job_id)
                elif job_id in _failed_jobs:
                    logger.info(f"check_status: job_id={job_id} → FAILED (in-memory)")
                    return _failed_jobs.pop(job_id)
                else:
                    logger.info(f"check_status: job_id={job_id} → RUNNING (background thread active)")
                    return {"status": "RUNNING"}

        else:
            raise ValueError(f"Unknown action: {action}")

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
