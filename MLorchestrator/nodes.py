import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional

from a2a_client import A2AClient
from state import MLorchestratorState
from logging_config import A2ACallLogger

logger = logging.getLogger(__name__)

def _get_call_logger(state: MLorchestratorState) -> A2ACallLogger:
    """Create or get A2ACallLogger for tracking interactions."""
    output_folder = state.get("output_folder", "./output")
    return A2ACallLogger(output_folder)

# ─── Node 1: call_perception_agent ──────────────────────────────────────────

def call_perception_agent(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Calls the external Perception Agent to analyze the input task.
    """
    logger.info("=" * 60)
    logger.info("  NODE: call_perception_agent")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    perception_url = a2a_cfg.get("perception_url", "http://localhost:8020")

    client = A2AClient(perception_url, agent_name="PerceptionAgent")
    client.call_logger = _get_call_logger(state)

    payload = {
        "input_data_folder": state.get("input_data_folder", ""),
        "output_folder": state.get("output_folder", ""),
        "user_input": state.get("user_input", ""),
        "config": config,
    }

    result = client.send_task_sync("analyze_task", payload)

    data_prompt = result.get("data_prompt", "")
    task_description = result.get("task_description", "")
    selected_tools = result.get("selected_tools", [])
    current_tool = result.get("current_tool", "")
    tool_prompt = result.get("tool_prompt", "")
    input_data_folder = result.get("input_data_folder") or state.get("input_data_folder", "")

    if not selected_tools:
        raise ValueError(
            "No ML tools were selected by the Perception Agent. "
            "Please ensure that the input data path is accessible to the Perception Agent "
            "(e.g., using an S3 URI like 's3://bucket/path' instead of a local host path like '/home/administrator/...'), "
            "and that the task description is valid."
        )

    logger.info(f"Perception successful.")
    logger.info(f"  Task Description: {task_description[:150]}...")
    logger.info(f"  Selected Tools: {selected_tools}")
    logger.info(f"  Current Tool: {current_tool}")

    return {
        "data_prompt": data_prompt,
        "task_description": task_description,
        "selected_tools": selected_tools,
        "current_tool": current_tool,
        "tool_prompt": tool_prompt,
        "input_data_folder": input_data_folder,
    }

# ─── Node 2: init_mcts ───────────────────────────────────────────────────────

def init_mcts(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Initializes the MCTS tree via mcts_handler Bedrock Agent.
    """
    logger.info("=" * 60)
    logger.info("  NODE: init_mcts")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    max_iter = state.get("max_iterations") or config.get("mcts", {}).get("max_iterations", 10)
    s3_bucket = state.get("s3_bucket") or config.get("s3_bucket") or os.environ.get("S3_BUCKET_NAME")

    payload = {
        "action": "init",
        "selected_tools": state.get("selected_tools", []),
        "config": config,
        "max_iterations": max_iter,
        "s3_bucket": s3_bucket,
    }

    if state.get("perception_results"):
        payload["perception_results"] = state["perception_results"]
    else:
        payload["perception_results"] = {
            "selected_tools": state.get("selected_tools", []),
            "task_description": state.get("task_description", ""),
            "data_prompt": state.get("data_prompt", ""),
            "tool_prompt": state.get("tool_prompt", ""),
            "current_tool": state.get("current_tool", "")
        }

    res = client.send_task_sync("init", payload)

    return {
        "mcts_tree": res,
        "iteration": 0,
        "max_iterations": max_iter,
        "is_complete": False,
        "all_error_analyses": [],
        "best_score": res.get("best_score"),
        "best_code": res.get("best_code", ""),
        "best_node_id": res.get("best_node_id")
    }

# ─── Node 3: select_node ─────────────────────────────────────────────────────

def select_node(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Traverses the MCTS tree via mcts_handler Bedrock Agent to select a leaf node.
    """
    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    iteration = state.get("iteration", 0)
    logger.info("=" * 60)
    logger.info(f"  NODE: select_node (Iteration {iteration}/{state.get('max_iterations')})")
    logger.info("=" * 60)

    payload = {
        "action": "select",
        "mcts_tree": state.get("mcts_tree")
    }

    res = client.send_task_sync("select", payload)

    node_id = res.get("node_id")
    is_complete = res.get("is_complete", False)
    
    if iteration >= state.get("max_iterations", 10):
        logger.info("Max iterations reached. Finalizing.")
        is_complete = True

    logger.info(f"Selected Node {node_id} (stage={res.get('stage')}, depth={res.get('depth')}, is_complete={is_complete})")

    current_selection = {
        "node_id": node_id,
        "stage": res.get("stage", "root"),
        "depth": res.get("depth", 0),
        "is_complete": is_complete,
        "current_tool": res.get("current_tool", ""),
        "parent_context": res.get("parent_context", {})
    }

    return {
        "current_selection": current_selection,
        "node_id": node_id,
        "stage": res.get("stage"),
        "depth": res.get("depth"),
        "current_tool": res.get("current_tool", ""),
        "is_complete": is_complete
    }

# ─── Node 4: expand_node ─────────────────────────────────────────────────────

def expand_node(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Creates a new child node in the MCTS tree via mcts_handler Bedrock Agent.
    """
    if state.get("is_complete"):
        return {}

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    logger.info("=" * 60)
    logger.info(f"  NODE: expand_node (Parent Node={state.get('node_id')})")
    logger.info("=" * 60)

    payload = {
        "action": "expand",
        "mcts_tree": state.get("mcts_tree"),
        "current_selection": state.get("current_selection")
    }

    res = client.send_task_sync("expand", payload)
    mcts_tree = res.get("mcts_tree", {})
    current_selection = res.get("current_selection", {})

    logger.info(
        f"Created child Node {current_selection.get('node_id')} "
        f"(stage={current_selection.get('stage')}, tool={current_selection.get('current_tool')}, depth={current_selection.get('depth')})"
    )

    return {
        "mcts_tree": mcts_tree,
        "current_selection": current_selection,
        "node_id": current_selection.get("node_id"),
        "stage": current_selection.get("stage"),
        "current_tool": current_selection.get("current_tool"),
        "depth": current_selection.get("depth")
    }

# ─── Node 5: call_memory_agent ───────────────────────────────────────────────

def call_memory_agent(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Calls the external Memory Agent to retrieve relevant tutorials/guidelines.
    """
    if state.get("is_complete"):
        return {}

    logger.info("=" * 60)
    logger.info("  NODE: call_memory_agent")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    memory_url = a2a_cfg.get("memory_url", "http://localhost:8021")

    client = A2AClient(memory_url, agent_name="MemoryAgent")
    client.call_logger = _get_call_logger(state)

    all_error_analyses = state.get("all_error_analyses")
    if not all_error_analyses and state.get("mcts_tree"):
        all_error_analyses = state["mcts_tree"].get("all_error_analyses", [])
    if not all_error_analyses:
        all_error_analyses = []

    payload = {
        "task_description": state.get("task_description", ""),
        "data_prompt": state.get("data_prompt", ""),
        "user_input": state.get("user_input", ""),
        "current_tool": state.get("current_tool", ""),
        "all_error_analyses": all_error_analyses,
        "config": config,
        "output_folder": state.get("output_folder", ""),
    }

    try:
        result = client.send_task_sync("retrieve_tutorials", payload)
        tutorial_prompt = result.get("tutorial_prompt", "")
        logger.info(f"Memory Agent retrieved tutorials successfully ({len(tutorial_prompt)} chars)")
        return {
            "tutorial_prompt": tutorial_prompt,
            "semantic_results": {"tutorial_prompt": tutorial_prompt}
        }
    except Exception as e:
        logger.error(f"Failed to communicate with Memory Agent: {e}")
        return {
            "tutorial_prompt": "Fallback: Standard modeling instructions",
            "semantic_results": {"tutorial_prompt": "Fallback: Standard modeling instructions"}
        }

# ─── Node 6: call_coding_agent ───────────────────────────────────────────────

def call_coding_agent(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Calls the external Coding Agent to write, run, and evaluate ML code using polling.
    """
    if state.get("is_complete"):
        return {}

    logger.info("=" * 60)
    logger.info("  NODE: call_coding_agent")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    coding_url = a2a_cfg.get("coding_url", "http://localhost:8022")

    client = A2AClient(coding_url, agent_name="CodingAgent")
    client.call_logger = _get_call_logger(state)

    # Generate a session ID to pin all calls (generate_and_run + all check_status
    # polls) to the same warm coder agent container. This avoids cold starts on
    # every poll and routes each call directly to the session where the background
    # polling thread is running.
    coder_session_id = str(uuid.uuid4())
    logger.info(f"Coder agent session_id for this node: {coder_session_id}")

    parent_ctx = state.get("current_selection", {}).get("parent_context", {})
    parent_code = parent_ctx.get("parent_code", "")
    parent_bash = parent_ctx.get("parent_bash", "")
    parent_error = parent_ctx.get("parent_error", "")

    all_error_analyses = state.get("all_error_analyses")
    if not all_error_analyses and state.get("mcts_tree"):
        all_error_analyses = state["mcts_tree"].get("all_error_analyses", [])
    if not all_error_analyses:
        all_error_analyses = []

    payload = {
        "action": "generate_and_run",
        "task_description": state.get("task_description", ""),
        "data_prompt": state.get("data_prompt", ""),
        "user_input": state.get("user_input", ""),
        "current_tool": state.get("current_tool", ""),
        "tool_prompt": state.get("tool_prompt", ""),
        "tutorial_prompt": state.get("tutorial_prompt", ""),
        "all_error_analyses": all_error_analyses,
        "config": config,
        "output_folder": state.get("output_folder", ""),
        "node_id": state.get("node_id", 0),
        "stage": state.get("stage", "evolve"),
        "iteration": state.get("iteration", 0),
        "parent_code": parent_code,
        "parent_bash": parent_bash,
        "parent_error": parent_error,
        "previous_python_code": parent_code,
        "previous_bash_script": parent_bash,
    }

    try:
        # Step 1: Initiate code generation and execution launch
        logger.info("Requesting Coder Agent to initiate asynchronous training task...")
        result = client.send_task_sync("generate_and_run", payload, session_id=coder_session_id)
        
        status = result.get("status")
        if status != "ACCEPTED":
            error_msg = result.get("error", "Task was not accepted by Coder Agent")
            raise RuntimeError(f"Coder Agent rejected invocation: {error_msg}")
            
        job_id = result.get("job_id")
        logger.info(f"Task accepted by Coder Agent. Job ID: {job_id}. Starting status polling...")
        
        # Step 2: Poll status loop
        poll_interval = 30  # seconds
        elapsed_time = 0
        while True:
            time.sleep(poll_interval)
            elapsed_time += poll_interval
            
            logger.info(f"Polling Coder Agent status (Elapsed: {elapsed_time}s)...")
            poll_payload = {
                "action": "check_status",
                "job_id": job_id,
                "task_description": state.get("task_description", ""),
                "data_prompt": state.get("data_prompt", ""),
                "config": config,
            }
            poll_result = client.send_task_sync("check_status", poll_payload, session_id=coder_session_id)
            poll_status = poll_result.get("status")
            
            if poll_status == "RUNNING":
                logger.info(f"Job {job_id} is still running inside the sandbox.")
            elif poll_status == "COMPLETED":
                python_code = poll_result.get("python_code", "")
                bash_script = poll_result.get("bash_script", "")
                stdout = poll_result.get("stdout", "")
                stderr = poll_result.get("stderr", "")
                decision = poll_result.get("decision", "FIX")
                error_summary = poll_result.get("error_summary", "")
                validation_score = poll_result.get("validation_score")
                error_analysis = poll_result.get("error_analysis", "")

                if validation_score is not None:
                    try:
                        validation_score = float(validation_score)
                    except ValueError:
                        logger.warning(f"Coding Agent returned non-numeric validation_score: {validation_score}")
                        validation_score = None

                logger.info(f"Background job completed. Decision: {decision}, Score: {validation_score}")
                
                return {
                    "coding_results": {
                        "python_code": python_code,
                        "bash_script": bash_script,
                        "stdout": stdout,
                        "stderr": stderr,
                        "decision": decision,
                        "error_summary": error_summary,
                        "validation_score": validation_score,
                        "error_analysis": error_analysis,
                        "error_message": error_summary if error_summary else stderr
                    },
                    "python_code": python_code,
                    "bash_script": bash_script,
                    "stdout": stdout,
                    "stderr": stderr,
                    "decision": decision,
                    "error_summary": error_summary,
                    "validation_score": validation_score,
                    "error_analysis": error_analysis,
                    "error_message": error_summary if error_summary else stderr
                }
            elif poll_status == "FAILED":
                error_msg = poll_result.get("error", "Unknown background execution failure")
                logger.error(f"Background job failed: {error_msg}")
                return {
                    "coding_results": {
                        "python_code": "",
                        "bash_script": "",
                        "stdout": "",
                        "stderr": error_msg,
                        "decision": "FIX",
                        "error_summary": error_msg,
                        "validation_score": None,
                        "error_analysis": error_msg,
                        "error_message": error_msg
                    },
                    "python_code": "",
                    "bash_script": "",
                    "stdout": "",
                    "stderr": error_msg,
                    "decision": "FIX",
                    "error_summary": error_msg,
                    "validation_score": None,
                    "error_analysis": error_msg,
                    "error_message": error_msg
                }
            else:
                logger.warning(f"Unknown status returned during polling: {poll_status}")
                
    except Exception as e:
        logger.error(f"Failed during communication with Coding Agent: {e}")
        return {
            "coding_results": {
                "python_code": "",
                "bash_script": "",
                "stdout": "",
                "stderr": str(e),
                "decision": "FIX",
                "error_summary": f"Communication error: {e}",
                "validation_score": None,
                "error_analysis": f"Orchestrator communication failure: {e}",
                "error_message": str(e)
            },
            "python_code": "",
            "bash_script": "",
            "stdout": "",
            "stderr": str(e),
            "decision": "FIX",
            "error_summary": f"Communication error: {e}",
            "validation_score": None,
            "error_analysis": f"Orchestrator communication failure: {e}",
            "error_message": str(e)
        }

# ─── Node 7: update_node ─────────────────────────────────────────────────────

def update_node(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Parses execution results from Coding Agent and updates tree via mcts_handler Bedrock Agent.
    """
    if state.get("is_complete"):
        return {}

    logger.info("=" * 60)
    logger.info("  NODE: update_node")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    coding_results = state.get("coding_results", {})
    if not coding_results:
        coding_results = {
            "python_code": state.get("python_code", ""),
            "bash_script": state.get("bash_script", ""),
            "stdout": state.get("stdout", ""),
            "stderr": state.get("stderr", ""),
            "decision": state.get("decision", "FIX"),
            "validation_score": state.get("validation_score"),
            "error_analysis": state.get("error_analysis", ""),
            "error_message": state.get("error_message", "")
        }

    payload = {
        "action": "update",
        "mcts_tree": state.get("mcts_tree"),
        "current_selection": state.get("current_selection"),
        "coding_results": coding_results
    }

    res = client.send_task_sync("update", payload)
    mcts_tree = res.get("mcts_tree", {})
    current_selection = res.get("current_selection", {})

    best_score = mcts_tree.get("best_score")
    best_code = mcts_tree.get("best_code", "")
    best_node_id = mcts_tree.get("best_node_id")
    all_error_analyses = mcts_tree.get("all_error_analyses", [])

    logger.info(f"Updated Node via mcts_handler. Best score so far: {best_score}")

    return {
        "mcts_tree": mcts_tree,
        "current_selection": current_selection,
        "best_score": best_score,
        "best_code": best_code,
        "best_node_id": best_node_id,
        "all_error_analyses": all_error_analyses
    }

# ─── Node 8: backpropagate ───────────────────────────────────────────────────

def backpropagate(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Backpropagates statistics up the MCTS tree via mcts_handler Bedrock Agent.
    """
    if state.get("is_complete"):
        return {}

    logger.info("=" * 60)
    logger.info(f"  NODE: backpropagate [Node={state.get('node_id')}]")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    payload = {
        "action": "backpropagate",
        "mcts_tree": state.get("mcts_tree"),
        "current_selection": state.get("current_selection")
    }

    res = client.send_task_sync("backpropagate", payload)
    mcts_tree = res
    iteration = mcts_tree.get("iteration", 0)

    return {
        "mcts_tree": mcts_tree,
        "iteration": iteration
    }

# ─── Node 9: finalize_results ────────────────────────────────────────────────

def finalize_results(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Saves MCTS final logs, tree visualizations, and finalizes run folders.
    """
    import json
    import shutil

    logger.info("=" * 60)
    logger.info("  NODE: finalize_results")
    logger.info("=" * 60)

    config = state.get("config", {})
    a2a_cfg = config.get("a2a_agents", {})
    mcts_url = a2a_cfg.get("mcts_url", "http://localhost:8023")

    client = A2AClient(mcts_url, agent_name="MCTSHandler")
    client.call_logger = _get_call_logger(state)

    payload = {
        "action": "finalize",
        "mcts_tree": state.get("mcts_tree")
    }

    tree_viz = ""
    try:
        res = client.send_task_sync("finalize", payload)
        tree_viz = res.get("tree_visualization", "")
        status = res.get("status", {})

        best_node_id = status.get("best_node_id")
        output_folder = state.get("output_folder", "./output")
        os.makedirs(output_folder, exist_ok=True)

        # 1. Copy best run folder if it exists
        if best_node_id is not None and output_folder:
            best_node_dir = os.path.join(output_folder, f"node_{best_node_id}")
            best_run_dir = os.path.join(output_folder, "best_run")
            if os.path.exists(best_node_dir):
                try:
                    if os.path.exists(best_run_dir):
                        shutil.rmtree(best_run_dir)
                    shutil.copytree(best_node_dir, best_run_dir)
                    logger.info(f"Best run copied from node_{best_node_id} to best_run/")
                except Exception as e:
                    logger.error(f"Failed to copy best run folder: {e}")

        # 2. Save flat tree JSON
        mcts_tree = state.get("mcts_tree", {})
        tree_path = os.path.join(output_folder, "mcts_tree.json")
        try:
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(mcts_tree, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved MCTS tree JSON to: {tree_path}")
        except Exception as e:
            logger.error(f"Failed to save MCTS tree JSON: {e}")

        # 3. Write tree visualization text file
        tree_txt_path = os.path.join(output_folder, "mcts_tree.txt")
        try:
            with open(tree_txt_path, "w", encoding="utf-8") as f:
                f.write(tree_viz)
            logger.info(f"Saved MCTS tree visualization to {tree_txt_path}")
            logger.info(f"\n{tree_viz}")
        except Exception as e:
            logger.error(f"Failed to write MCTS tree visualization: {e}")

        # 4. Upload entire output_folder to S3 so it can be downloaded locally
        mcts_tree = state.get("mcts_tree", {})
        s3_uri = mcts_tree.get("s3_uri", "") if isinstance(mcts_tree, dict) else ""
        if s3_uri and output_folder:
            try:
                import boto3
                # Parse run_id from s3_uri: s3://mlzero-output/mlorchestrator/runs/{run_id}/mcts_tree.json
                s3_parts = s3_uri.replace("s3://", "").split("/", 1)
                bucket = s3_parts[0]
                # run_id is the folder just before mcts_tree.json
                key_parts = s3_parts[1].rsplit("/", 1)
                run_prefix = key_parts[0]  # e.g. mlorchestrator/runs/20260630_151155_0dff6381
                artifacts_prefix = f"{run_prefix}/artifacts"

                logger.info(f"Uploading output artifacts to s3://{bucket}/{artifacts_prefix}/ ...")
                s3_client = boto3.client("s3")
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        local_file = os.path.join(root, file)
                        rel_path = os.path.relpath(local_file, output_folder)
                        s3_key = f"{artifacts_prefix}/{rel_path}"
                        try:
                            s3_client.upload_file(local_file, bucket, s3_key)
                            logger.info(f"  Uploaded: {rel_path}")
                        except Exception as upload_err:
                            logger.error(f"  Failed to upload {rel_path}: {upload_err}")
                logger.info(f"S3 artifact upload complete: s3://{bucket}/{artifacts_prefix}/")
            except Exception as e:
                logger.error(f"Failed to upload output artifacts to S3: {e}")

    except Exception as e:
        logger.error(f"Failed to communicate with MCTS Handler during finalization: {e}")

    logger.info("Pipeline finalized successfully.")
    return {"is_complete": True, "tree_visualization": tree_viz}


def _create_sandbox_client(sandbox_url: str, read_timeout: int = 900, max_retries: int = 0):
    """Create a SandboxClient with custom timeout and retry settings."""
    import boto3
    from botocore.config import Config as BotoConfig

    from sandbox_client import SandboxClient

    sandbox = SandboxClient(sandbox_url)
    # Override the boto3 Lambda client with custom timeout/retry config
    sandbox.lambda_client = boto3.client(
        "lambda",
        region_name=sandbox.region_name,
        config=BotoConfig(
            read_timeout=read_timeout,
            connect_timeout=30,
            retries={'max_attempts': max_retries}
        )
    )
    return sandbox


def sync_s3_to_sandbox(state: MLorchestratorState) -> Dict[str, Any]:
    """
    Syncs the S3 input_data_folder into the sandbox if it is an S3 URI.
    Uses short timeouts for health checks and retries for transient failures.
    """
    logger.info("=" * 60)
    logger.info("  NODE: sync_s3_to_sandbox")
    logger.info("=" * 60)

    input_folder = state.get("input_data_folder", "")
    if not input_folder or not input_folder.startswith("s3://"):
        logger.info("Input folder is not an S3 URI. Skipping sync step.")
        return {"input_data_folder": input_folder}

    import hashlib
    import boto3

    path_hash = hashlib.md5(input_folder.encode("utf-8")).hexdigest()[:8]
    sandbox_sync_dir = f"/tmp/s3_data_{path_hash}"
    logger.info(f"Syncing S3 URI '{input_folder}' to sandbox path '{sandbox_sync_dir}'")

    sandbox_url = os.environ.get("SANDBOX_URL", "lambda:fame-sandbox-bastion")

    # ── Step 1: Quick health check with SHORT timeout (60s) and NO retries ──
    skip_sync = False
    try:
        check_sandbox = _create_sandbox_client(sandbox_url, read_timeout=60, max_retries=0)
        check_success, check_stdout, _ = check_sandbox.exec_shell_sync(
            command=f"test -d {sandbox_sync_dir} && find {sandbox_sync_dir} -type f | head -n 1",
            cwd="/home/gem/workspace"
        )
        if check_success and check_stdout.strip():
            logger.info(f"Sandbox sync directory '{sandbox_sync_dir}' already has files. Skipping S3 sync.")
            skip_sync = True
    except Exception as e:
        logger.warning(f"Directory check failed (will proceed with sync): {e}")

    if not skip_sync:
        # ── Step 2: Gather AWS credentials for the sandbox ──
        access_key = None
        secret_key = None
        token = None
        region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                frozen_creds = credentials.get_frozen_credentials()
                access_key = frozen_creds.access_key
                secret_key = frozen_creds.secret_key
                token = frozen_creds.token
        except Exception as e:
            logger.warning(f"Could not retrieve credentials from boto3 Session: {e}")

        env_prefix = ""
        if access_key and secret_key:
            env_prefix = f"AWS_ACCESS_KEY_ID={access_key} AWS_SECRET_ACCESS_KEY={secret_key} "
            if token:
                env_prefix += f"AWS_SESSION_TOKEN={token} "
            if region:
                env_prefix += f"AWS_DEFAULT_REGION={region} "

        sync_command = f"mkdir -p {sandbox_sync_dir} && {env_prefix}aws s3 sync {input_folder} {sandbox_sync_dir}"

        # ── Step 3: Execute S3 sync with retries and backoff ──
        max_retries = 3
        last_error = None
        for attempt in range(1, max_retries + 1):
            logger.info(f"Executing aws s3 sync inside sandbox (attempt {attempt}/{max_retries})...")
            try:
                # Use 900s timeout for the actual sync (large datasets take time)
                sync_sandbox = _create_sandbox_client(sandbox_url, read_timeout=900, max_retries=0)
                success, stdout, stderr = sync_sandbox.exec_shell_sync(
                    command=sync_command,
                    cwd="/home/gem/workspace"
                )
                if success:
                    logger.info(f"Successfully synced S3 to sandbox at '{sandbox_sync_dir}'")
                    last_error = None
                    break
                else:
                    last_error = stderr
                    logger.error(f"S3 sync attempt {attempt} failed: {stderr}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"S3 sync attempt {attempt} exception: {e}")

            if attempt < max_retries:
                backoff = 15 * attempt
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)

        if last_error:
            raise RuntimeError(f"Unable to sync S3 to sandbox after {max_retries} attempts: {last_error}")

    return {"input_data_folder": sandbox_sync_dir}

