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
    
    import psutil
    process = psutil.Process()
    start_cpu = process.cpu_times()
    try:
        start_io = process.io_counters()
    except Exception:
        start_io = None
        
    t0 = time.time()
    result = await _graph.ainvoke(initial_state, config=langgraph_config)
    elapsed = time.time() - t0
    
    end_cpu = psutil.Process().cpu_times()
    active_cpu_s = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    wait_time_s = max(0, elapsed - active_cpu_s)

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
        "event_type": "psutil_metrics_graph",
        "graph_name": "coder_agent_run",
        "graph_e2e_s": round(elapsed, 4),
        "active_cpu_s": round(active_cpu_s, 4),
        "wait_time_s": round(wait_time_s, 4),
        "io_read_MB": round(io_read_mb, 4),
        "io_write_MB": round(io_write_mb, 4),
        "step_count": 3,
        "iteration_count": payload.get("iteration", 0)
    })

    return result


def _background_poll_and_evaluate(
    job_id: str,
    task_id: int,
    payload: Dict[str, Any],
    invocation_start_ms: int,
) -> None:
    """
    Background thread:
    1. Generates code and submits the job to the sandbox (_run_coder_core).
    2. Polls the sandbox every 30s until the training job completes.
    3. Runs LLM evaluation, stores results in memory, and completes the task.
    """
    logger.info(f"[BG] Background thread started for job_id={job_id}")
    try:
        # Step 1: Run LLM generation and submit job to sandbox
        logger.info(f"[BG] Initiating code generation & sandbox submission for job_id={job_id}...")
        asyncio.run(_run_coder_core(payload, invocation_start_ms))
        logger.info(f"[BG] Sandbox job for job_id={job_id} successfully launched.")

        # Step 2: Poll sandbox status until completion
        poll_interval = 30
        check_payload = {
            "job_id": job_id,
            "config": payload.get("config", {}),
            "task_description": payload.get("task_description", ""),
            "data_prompt": payload.get("data_prompt", ""),
        }

        elapsed = 0
        max_time = 1860  # 30 mins + 1 minute buffer

        while True:
            time.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed > max_time:
                logger.error(f"[BG] Sandbox polling timed out for job_id={job_id} after {elapsed}s.")
                with _job_store_lock:
                    _failed_jobs[job_id] = {"status": "FAILED", "error": "Sandbox execution exceeded 30-minute time limit"}
                break

            logger.info(f"[BG] Polling sandbox for job_id={job_id}...")
            try:
                result = asyncio.run(_check_coder_status(check_payload))
            except Exception as poll_exc:
                logger.error(f"[BG] Sandbox poll error for job_id={job_id}: {poll_exc}")
                result = {"status": "RUNNING"}  # optimistic: keep retrying

            status = result.get("status")
            if status == "RUNNING":
                logger.info(f"[BG] job_id={job_id} still running.")
                continue
            elif status == "COMPLETED":
                logger.info(f"[BG] job_id={job_id} COMPLETED. Storing result.")
                with _job_store_lock:
                    _completed_jobs[job_id] = result
                break
            else:  # FAILED or unexpected
                logger.warning(f"[BG] job_id={job_id} FAILED: {result.get('error', '')}")
                with _job_store_lock:
                    _failed_jobs[job_id] = result
                break
    except Exception as exc:
        logger.error(f"[BG] Fatal error in background thread for job_id={job_id}: {exc}", exc_info=True)
        with _job_store_lock:
            _failed_jobs[job_id] = {"status": "FAILED", "error": str(exc)}
    finally:
        app.complete_async_task(task_id)
        logger.info(f"[BG] Background thread finished for job_id={job_id}. Session released.")


async def _check_coder_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = payload.get("job_id")
    if not job_id:
        raise ValueError("Missing 'job_id' for check_status action.")
        
    config = payload.get("config", {})
    sandbox_url = config.get("sandbox_url") or os.environ.get("SANDBOX_URL")
    sandbox = SandboxClient(sandbox_url)
    
    iter_folder = f"/home/gem/workspace/{job_id}"
    
    # 1. Check if exit_code.txt exists (job finished)
    success, stdout_ec, stderr_ec = await sandbox.exec_shell(
        "test -f exit_code.txt && cat exit_code.txt", cwd=iter_folder
    )
    
    if not success:
        # exit_code.txt does not exist yet. Check if the job is still running.
        # Strategy: Try multiple detection methods in order of reliability.
        
        # Method 1: Check run.pid if it exists
        pid_alive = False
        success_pid, pid_out, _ = await sandbox.exec_shell(
            "test -f run.pid && cat run.pid", cwd=iter_folder
        )
        if success_pid and pid_out.strip():
            pid = pid_out.strip()
            alive_check, _, _ = await sandbox.exec_shell(
                f"kill -0 {pid} 2>/dev/null", cwd=""
            )
            if alive_check:
                pid_alive = True
        
        # Method 2: Check if execution_script.sh is running via pgrep
        if not pid_alive:
            success_pgrep, pgrep_out, _ = await sandbox.exec_shell(
                f"pgrep -f 'execution_script.sh' || pgrep -f '{iter_folder}'", cwd=""
            )
            if success_pgrep and pgrep_out.strip():
                pid_alive = True
        
        # Method 3: Check if stderr.log is actively growing (take two snapshots)
        if not pid_alive:
            success_sz1, sz1_out, _ = await sandbox.exec_shell(
                "stat -c %s stderr.log 2>/dev/null || echo 0", cwd=iter_folder
            )
            if success_sz1:
                import asyncio as _asyncio
                await _asyncio.sleep(2)
                success_sz2, sz2_out, _ = await sandbox.exec_shell(
                    "stat -c %s stderr.log 2>/dev/null || echo 0", cwd=iter_folder
                )
                if success_sz2:
                    try:
                        sz1 = int(sz1_out.strip())
                        sz2 = int(sz2_out.strip())
                        if sz2 > sz1:
                            pid_alive = True
                    except ValueError:
                        pass
        
        if pid_alive:
            return {"status": "RUNNING"}
        else:
            # All detection methods failed — job likely died
            stderr_content = ""
            try:
                stderr_content = await sandbox.read_file(f"{iter_folder}/stderr.log")
            except Exception:
                pass
            return {
                "status": "FAILED",
                "error": f"Background execution died unexpectedly. Stderr: {stderr_content}"
            }
            
    # exit_code.txt exists. Retrieve logs and evaluate.
    exit_code_str = stdout_ec.strip()
    logger.info(f"Background job {job_id} finished with exit code {exit_code_str}. Evaluating results...")
    
    # Read stdout and stderr logs
    stdout_content = ""
    stderr_content = ""
    try:
        stdout_content = await sandbox.read_file(f"{iter_folder}/stdout.log")
    except Exception as e:
        logger.warning(f"Could not read stdout.log: {e}")
        
    try:
        stderr_content = await sandbox.read_file(f"{iter_folder}/stderr.log")
    except Exception as e:
        logger.warning(f"Could not read stderr.log: {e}")
        
    # Read python and bash scripts
    python_code = ""
    try:
        python_code = await sandbox.read_file(f"{iter_folder}/generated_code.py")
    except Exception as e:
        logger.warning(f"Could not read generated_code.py: {e}")
        
    bash_script = ""
    try:
        bash_script = await sandbox.read_file(f"{iter_folder}/execution_script.sh")
    except Exception as e:
        logger.warning(f"Could not read execution_script.sh: {e}")
        
    # Initialize LLM config for evaluation
    llm_config = config.get("llm", {}).copy()
    if "model" not in llm_config:
        llm_config["model"] = os.environ.get("LLM_MODEL", "gpt-4o")
        
    from coder_agent.agent import evaluate_execution_results
    
    eval_result = await evaluate_execution_results(
        llm_config=llm_config,
        task_description=payload.get("task_description", ""),
        data_prompt=payload.get("data_prompt", ""),
        python_code=python_code,
        stdout=stdout_content,
        stderr=stderr_content,
    )
    
    return {
        "status": "COMPLETED",
        "python_code": python_code,
        "bash_script": bash_script,
        "stdout": stdout_content,
        "stderr": stderr_content,
        "decision": eval_result.get("decision", "FIX"),
        "validation_score": eval_result.get("validation_score"),
        "error_message": eval_result.get("error_message", ""),
        "error_analysis": eval_result.get("error_analysis", "")
    }


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    Main AgentCore app entrypoint.

    Actions:
    - generate_and_run: Generate code, launch sandbox training job, register
      async task (HealthyBusy), spawn background polling thread, return ACCEPTED.
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
