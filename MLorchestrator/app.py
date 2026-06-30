import asyncio
import logging
import os
import sys
import threading
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables
_curr_dir = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(_curr_dir / ".env")
except ImportError:
    pass

# Ensure package root is in sys.path
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from graph import build_orchestrator_graph
from logging_config import configure_logging

# ─── Setup Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlorchestrator.app")

# ─── Config Loading Helpers ────────────────────────────────────────────────────
def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges dict2 into dict1."""
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML config file. Returns empty dict if file not found or invalid."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        logger.error(f"Error loading YAML config from {config_path}: {e}")
        return {}

def load_config(config_path: str = None) -> dict:
    """Loads configuration merging default and optional override."""
    default_path = _curr_dir / "config.yaml"
    config = load_yaml_config(str(default_path))
    if config_path and Path(config_path).exists():
        user_config = load_yaml_config(config_path)
        config = deep_merge(config, user_config)
    return config

# ─── AgentCore App Initialization ──────────────────────────────────────────────
app = BedrockAgentCoreApp()

# In-memory stores for background run executions
_completed_runs: Dict[str, Dict[str, Any]] = {}
_failed_runs: Dict[str, Dict[str, Any]] = {}
_runs_lock = threading.Lock()

# ─── Background Thread Execution Loop ───────────────────────────────────────────
def _background_mcts_thread(
    run_id: str,
    task_id: int,
    initial_state: Dict[str, Any],
    output_path: Optional[Path],
    effective_max_iter: int,
    config: Dict[str, Any],
    input_data_folder: Optional[str],
    user_input: str,
) -> None:
    """
    Runs the LangGraph MCTS pipeline synchronously in a background thread
    and keeps the AgentCore container alive by holding the async task.
    """
    logger.info(f"[BG] Background MCTS run {run_id} started.")
    start_time = time.time()
    try:
        # Build LangGraph and execute search loop
        graph = build_orchestrator_graph()
        result = graph.invoke(initial_state)
        elapsed = time.time() - start_time

        # Extract values for completion dictionary
        mcts_tree = result.get("mcts_tree", {})
        best_score = result.get("best_score")
        if best_score is None and mcts_tree:
            best_score = mcts_tree.get("best_validation_score")
        best_node_id = result.get("best_node_id")
        if best_node_id is None and mcts_tree:
            best_node_id = mcts_tree.get("best_node_id")
        iteration = result.get("iteration", 0)
        if iteration == 0 and mcts_tree:
            iteration = mcts_tree.get("iteration", 0)

        completed_result = {
            "status": "COMPLETED",
            "best_score": best_score,
            "best_node_id": best_node_id,
            "total_iterations": iteration,
            "elapsed_time": elapsed,
            "output_folder": str(output_path) if output_path else None,
            "is_complete": result.get("is_complete", False),
            "mcts_tree": result.get("mcts_tree"),
            "tree_visualization": result.get("tree_visualization", "")
        }

        # Write ml_orchestrator.json to output directory
        if output_path:
            import json
            orchestrator_log = {
                "timestamp": datetime.now().isoformat(),
                "skill": "orchestrate",
                "input": {
                    "input_data_folder": input_data_folder,
                    "user_input": user_input,
                    "max_iterations": effective_max_iter,
                    "config": config,
                },
                "output": {
                    "best_score": best_score,
                    "best_node_id": best_node_id,
                    "total_iterations": iteration,
                    "is_complete": result.get("is_complete", False),
                },
                "time_taken_seconds": round(elapsed, 3),
            }
            try:
                with open(output_path / "ml_orchestrator.json", "w", encoding="utf-8") as f:
                    json.dump([orchestrator_log], f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to write MLorchestrator log: {e}")

        logger.info(f"[BG] Background MCTS run {run_id} finished successfully. Best Score: {best_score}")
        with _runs_lock:
            _completed_runs[run_id] = completed_result

    except Exception as exc:
        logger.error(f"[BG] Background MCTS run {run_id} failed: {exc}", exc_info=True)
        with _runs_lock:
            _failed_runs[run_id] = {
                "status": "FAILED",
                "error": str(exc),
            }
    finally:
        # Revert container status to HEALTHY so it can idle out eventually
        app.complete_async_task(task_id)
        logger.info(f"[BG] Background task task_id={task_id} completed. Session released.")


@app.entrypoint
def handle(payload: dict) -> dict:
    """
    AgentCore entrypoint for MLorchestrator.
    Actions:
    - start_run (default): Sets up run directory (run_N), registers the async
      task (HealthyBusy status), spawns background thread, returns run_id immediately.
    - check_status: In-memory lookup of the status of the requested run_id.
    """
    action = payload.get("action", "start_run")
    logger.info(f"Orchestrator Agent Invoked: action={action}")

    if action == "start_run":
        run_id = str(uuid.uuid4())[:8]
        input_data_folder = payload.get("input_data_folder")
        user_input = payload.get("user_input", "")
        max_iterations = payload.get("max_iterations")
        output_folder = payload.get("output_folder")
        s3_uri = payload.get("s3_uri")
        s3_bucket = payload.get("s3_bucket")

        # Load & merge configurations
        config = load_config(payload.get("config_path"))
        payload_config = payload.get("config", {})
        if payload_config and isinstance(payload_config, dict):
            config = deep_merge(config, payload_config)

        # Override downstream agent ARNs from env vars
        a2a_cfg = config.setdefault("a2a_agents", {})
        for env_var, key in [
            ("PERCEPTION_AGENT_ARN", "perception_url"),
            ("SEMANTIC_AGENT_ARN", "memory_url"),
            ("MEMORY_AGENT_ARN", "memory_url"),
            ("CODING_AGENT_ARN", "coding_url"),
            ("MCTS_HANDLER_ARN", "mcts_url"),
        ]:
            val = os.environ.get(env_var)
            if val:
                a2a_cfg[key] = val

        # Configure max MCTS iterations
        mcts_config = config.get("mcts", {})
        if max_iterations is not None:
            mcts_config["max_iterations"] = max_iterations
        config["mcts"] = mcts_config
        effective_max_iter = mcts_config.get("max_iterations", 10)

        # Normalize input folder if local
        normalized_input = None
        if input_data_folder:
            if input_data_folder.startswith(("s3://", "http://", "https://")):
                normalized_input = input_data_folder
            else:
                normalized_input = str(Path(input_data_folder).resolve())

        # Determine output path with standard run_N naming convention
        output_path = None
        if output_folder:
            p = Path(output_folder)
            output_path = p if p.is_absolute() else (_curr_dir.parent / p).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        elif normalized_input:
            runs_dir = _curr_dir.parent / "runs"
            runs_dir.mkdir(exist_ok=True)

            # Determine next run_N index
            max_num = 0
            for child in runs_dir.iterdir():
                if child.is_dir() and child.name.startswith("run_"):
                    try:
                        num = int(child.name.split("_")[-1])
                        if num > max_num:
                            max_num = num
                    except (IndexError, ValueError):
                        pass
            output_path = runs_dir / f"run_{max_num + 1}"
            output_path.mkdir(parents=True, exist_ok=True)

        # Configure Logging
        if output_path:
            configure_logging(output_dir=str(output_path), verbosity=payload.get("verbosity", 2))

        # Setup initial state
        initial_state = {
            "s3_uri": s3_uri,
            "s3_bucket": s3_bucket,
            "config": config,
            "max_iterations": effective_max_iter,
            "user_input": user_input,
        }
        if normalized_input:
            initial_state["input_data_folder"] = normalized_input
        if output_path:
            initial_state["output_folder"] = str(output_path)

        # Register async task so container reports HealthyBusy
        task_id = app.add_async_task("mcts_run", metadata={"run_id": run_id})
        logger.info(f"Registered async task_id={task_id} for run_id={run_id}.")

        # Spawn background execution thread
        bg_thread = threading.Thread(
            target=_background_mcts_thread,
            args=(
                run_id,
                task_id,
                initial_state,
                output_path,
                effective_max_iter,
                config,
                normalized_input,
                user_input,
            ),
            daemon=True,
            name=f"orchestrator-{run_id}",
        )
        bg_thread.start()

        return {
            "status": "ACCEPTED",
            "run_id": run_id,
        }

    elif action == "check_status":
        run_id = payload.get("run_id")
        if not run_id:
            raise ValueError("Missing required 'run_id' for check_status action.")

        with _runs_lock:
            if run_id in _completed_runs:
                logger.info(f"check_status: run_id={run_id} completed. Returning results.")
                return _completed_runs.pop(run_id)
            elif run_id in _failed_runs:
                logger.info(f"check_status: run_id={run_id} failed. Returning error.")
                return _failed_runs.pop(run_id)
            else:
                return {"status": "RUNNING"}

    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    app.run()
