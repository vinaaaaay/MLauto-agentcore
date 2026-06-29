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
    Executes code generation or checks the status of background task execution.
    """
    invocation_start_ms = int(time.time() * 1000)
    ctx.init_from_payload(payload)
    
    action = payload.get("action", "generate_and_run")
    logger.info(f"Coder Agent Invoked: action={action}")

    try:
        if action == "generate_and_run":
            result = asyncio.run(_run_coder_core(payload, invocation_start_ms))
            
            emit_event(metric_logger, {
                **ctx.snapshot(),
                "event_type": "invocation",
                "status": "COMPLETED",
                "invocation_start_ms": invocation_start_ms,
                "total_ms": int(time.time() * 1000) - invocation_start_ms,
            })

            return {
                "status": "ACCEPTED",
                "job_id": result.get("job_id", ""),
                "python_code": result.get("python_code", ""),
                "bash_script": result.get("bash_script", ""),
            }
        elif action == "check_status":
            result = asyncio.run(_check_coder_status(payload))
            return result
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
