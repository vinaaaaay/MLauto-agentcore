import os
import json
import time
import uuid
import logging
import operator
import sys
import psutil
import functools
import inspect
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Any, Callable
import requests
from dotenv import load_dotenv

# Load .env file from directory of orchestrator.py
load_dotenv(Path(__file__).resolve().parent / ".env")

from langgraph.graph import StateGraph, START, END

from common.metrics_context import MetricsContext
from common.metrics_emitter import emit_event

logger = logging.getLogger(__name__)

# Metrics configuration
ctx = MetricsContext(agent_id="mlorchestrator")
metric_logger = logging.getLogger("agent_metrics")

# SigV4 credential resolver (kept region for SQS/S3 boto3 clients if needed)
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

# ── SQS Status Publisher ──────────────────────────────────────────────────────
RESULT_QUEUE_URL = os.environ.get("RESULT_QUEUE_URL", "")
_sqs_client = None

def _get_sqs_client():
    global _sqs_client
    if _sqs_client is None:
        _sqs_client = boto3.client("sqs", region_name=AWS_REGION)
    return _sqs_client

def publish_status(session_id: str, node_name: str, status: str,
                   duration_s: float = None, error: str = None,
                   extra: dict = None):
    """Publish a timestamped status update to the client's SQS queue."""
    if not RESULT_QUEUE_URL:
        return
    message = {
        "session_id": session_id,
        "node_name": node_name,
        "status": status,                           # STARTING | COMPLETED | FAILED | ERROR
        "timestamp": datetime.utcnow().isoformat(),
    }
    if duration_s is not None:
        message["duration_s"] = round(duration_s, 4)
    if error:
        message["error"] = error
    if extra:
        message.update(extra)
    try:
        import boto3
        _get_sqs_client().send_message(
            QueueUrl=RESULT_QUEUE_URL,
            MessageBody=json.dumps(message, default=str),
        )
    except Exception as e:
        logger.warning(f"SQS publish failed: {e}")

# Function URL / AgentCore service endpoints
SANDBOX_URL = os.environ.get("SANDBOX_S3_SYNC_URL", "")
PERCEPTION_URL = os.environ.get("PERCEPTION_AGENT_URL", "")
SEMANTIC_URL = os.environ.get("SEMANTIC_AGENT_URL", "")
CODER_URL = os.environ.get("CODER_AGENT_URL", "")
MCTS_URL = os.environ.get("MCTS_HANDLER_URL", "")

HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "900"))

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class OrchestratorState(TypedDict):
    """Represents the global state passed through the LangGraph workflow."""
    # Inputs
    s3_uri: Optional[str]
    input_data_folder: str
    user_input: str
    config: dict
    max_iterations: int
    s3_bucket: Optional[str]

    # Execution State
    perception_results: dict
    mcts_tree: dict
    current_selection: dict
    semantic_results: dict
    coding_results: dict
    update_result: dict
    final_outcome: dict
    
    # Telemetry (uses operator.add to append/increment across nodes automatically)
    telemetry_logs: Annotated[List[dict], operator.add]
    call_index: Annotated[int, operator.add]


# ---------------------------------------------------------
# 1.5 Node Metrics and Logging Decorator
# ---------------------------------------------------------
def orchestrator_node_metrics(node_name: str):
    """
    Decorator for orchestrator LangGraph node functions.
    Handles:
        - Setting node name in tracing context.
        - Emitting a 'node_input' event with the incoming state.
        - Measuring time and peak memory (psutil metrics).
        - Capturing errors and logging a 'node_error' event.
        - Emitting a 'node_output' event with the state update dict returned.
        - Emitting a 'psutil_metrics_node' event.
    """
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(state: OrchestratorState, *args, **kwargs):
                ctx.node_name.set(node_name)
                
                # Log input
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "node_input",
                    "node_name": node_name,
                    "input": dict(state),
                })
                
                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()
                
                try:
                    result = await func(state, *args, **kwargs)
                except Exception as e:
                    # Log error
                    emit_event(metric_logger, {
                        **ctx.snapshot(),
                        "event_type": "node_error",
                        "node_name": node_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    })
                    raise e
                
                # Log output/result
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "node_output",
                    "node_name": node_name,
                    "output": result,
                })
                
                # Log psutil metrics
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "psutil_metrics_node",
                    "node_name": node_name,
                    "node_e2e_s": round(time.time() - t0, 4),
                    "peak_RAM_GB": round(process.memory_info().rss / (1024**3), 4),
                })
                
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(state: OrchestratorState, *args, **kwargs):
                ctx.node_name.set(node_name)
                session_id = ctx.session_id.get()
                publish_status(session_id, node_name, "STARTING")
                
                # Log input
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "node_input",
                    "node_name": node_name,
                    "input": dict(state),
                })
                
                process = psutil.Process()
                initial_mem = process.memory_info().rss
                t0 = time.time()
                
                try:
                    result = func(state, *args, **kwargs)
                except Exception as e:
                    # Log error
                    emit_event(metric_logger, {
                        **ctx.snapshot(),
                        "event_type": "node_error",
                        "node_name": node_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    })
                    elapsed = time.time() - t0
                    publish_status(session_id, node_name, "ERROR", duration_s=elapsed, error=str(e))
                    raise e
                
                elapsed = time.time() - t0
                peak_mem = max(initial_mem, process.memory_info().rss)
                
                # Log output/result
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "node_output",
                    "node_name": node_name,
                    "output": result,
                })
                
                # Log psutil metrics
                emit_event(metric_logger, {
                    **ctx.snapshot(),
                    "event_type": "psutil_metrics_node",
                    "node_name": node_name,
                    "node_e2e_s": round(elapsed, 4),
                    "peak_RAM_GB": round(peak_mem / (1024**3), 4),
                })
                
                publish_status(session_id, node_name, "COMPLETED", duration_s=elapsed)
                return result
            return sync_wrapper
    return decorator


# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------

def invoke_http_utility(url: str, payload: dict, timeout: int = HTTP_TIMEOUT) -> dict:
    """POST a JSON payload to the URL and return the parsed response."""
    post_url = url
    # Ensure URL is directed to standard AgentCore /invocations route unless it specifies a known endpoint
    if not any(url.endswith(x) for x in ["/invocations", "/ping", "/retrieve_tutorials", "/sync_s3_to_sandbox"]):
        post_url = url.rstrip("/") + "/invocations"
        
    try:
        logger.info(f"Invoking URL: {post_url}")
        resp = requests.post(post_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"HTTP request to {post_url} timed out after {timeout}s")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error from {post_url}: {e.response.status_code} — {e.response.text[:500]}")
    except Exception as e:
        logger.error(f"Failed HTTP call to {post_url}: {e}")
        raise

def call_and_log(
    target: str,
    action: str,
    input_payload: dict,
    state: OrchestratorState,
    transport: str = "http",
) -> tuple[dict, dict]:
    """Wraps invoke_http_utility with telemetry generation."""
    # Propagate tracing context to the sub-agent
    lambda_payload = input_payload.copy() if input_payload is not None else {}
    if "session_id" not in lambda_payload:
        lambda_payload["session_id"] = ctx.session_id.get()
    if "tracing" not in lambda_payload:
        lambda_payload["tracing"] = ctx.child_context()

    current_index = state.get("call_index", 0) + 1
    call_start = time.time()
    call_start_iso = datetime.utcnow().isoformat()
    
    telemetry_entry = {
        "call_index": current_index,
        "target": target,
        "transport": "http",
        "action": action,
        "start_time": call_start_iso,
        "duration_seconds": 0.0,
        "status": "PENDING",
        "payload": lambda_payload,
        "response": None,
        "error": None
    }
    
    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type": "invocation_start",
        "target": target,
        "action": action,
        "transport": "http",
        "call_index": current_index,
        "timestamp": call_start_iso
    })
    
    try:
        res = invoke_http_utility(target, lambda_payload)
        duration = time.time() - call_start
        telemetry_entry.update({
            "duration_seconds": round(duration, 3),
            "status": "SUCCESS",
            "response": res
        })
        
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation_complete",
            "target": target,
            "action": action,
            "transport": "http",
            "call_index": current_index,
            "duration_seconds": round(duration, 3),
            "status": "SUCCESS"
        })
        return res, telemetry_entry
    except Exception as e:
        duration = time.time() - call_start
        telemetry_entry.update({
            "duration_seconds": round(duration, 3),
            "status": "FAILED",
            "error": str(e)
        })
        
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation_failed",
            "target": target,
            "action": action,
            "transport": "http",
            "call_index": current_index,
            "duration_seconds": round(duration, 3),
            "status": "FAILED",
            "error": str(e)
        })
        raise e


# ---------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------
@orchestrator_node_metrics(node_name="sync_s3_to_sandbox")
def sync_s3_to_sandbox(state: OrchestratorState):
    logger.info("Executing Sandbox Sync step...")
    payload = {
        "s3_uri": state.get("s3_uri"),
        "input_data_folder": state.get("input_data_folder"),
        "user_input": state.get("user_input"),
        "config": state.get("config")
    }
    res, log_entry = call_and_log(SANDBOX_URL, "sync_s3_to_sandbox", payload, state, transport="http")
    return {"telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="perception_agent")
def perception_agent(state: OrchestratorState):
    logger.info("Executing Perception Agent...")
    payload = {
        "input_data_folder": state["input_data_folder"],
        "user_input": state["user_input"],
        "config": state["config"]
    }
    res, log_entry = call_and_log(PERCEPTION_URL, "perception", payload, state, transport="http")
    
    perception_results = {
        "data_prompt": res.get("data_prompt"),
        "task_description": res.get("task_description"),
        "selected_tools": res.get("selected_tools"),
        "current_tool": res.get("current_tool"),
        "tool_prompt": res.get("tool_prompt"),
        "tutorial_prompt": res.get("tutorial_prompt")
    }
    return {"perception_results": perception_results, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="init_mcts")
def init_mcts(state: OrchestratorState):
    logger.info("Initializing MCTS tree...")
    payload = {
        "action": "init",
        "selected_tools": state["perception_results"].get("selected_tools", []),
        "config": state["config"],
        "max_iterations": state["max_iterations"]
    }
    res, log_entry = call_and_log(MCTS_URL, "init", payload, state, transport="http")
    return {"mcts_tree": res, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="select_node")
def select_node(state: OrchestratorState):
    logger.info("MCTS: Select Node...")
    payload = {
        "action": "select",
        "mcts_tree": state["mcts_tree"]
    }
    res, log_entry = call_and_log(MCTS_URL, "select", payload, state, transport="http")
    
    current_selection = {
        "node_id": res.get("node_id"),
        "stage": res.get("stage"),
        "depth": res.get("depth"),
        "is_complete": res.get("is_complete"),
        "current_tool": res.get("current_tool"),
        "parent_context": res.get("parent_context", {})
    }
    return {"current_selection": current_selection, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="expand_node")
def expand_node(state: OrchestratorState):
    logger.info("MCTS: Expand Node...")
    payload = {
        "action": "expand",
        "mcts_tree": state["mcts_tree"],
        "current_selection": state["current_selection"]
    }
    res, log_entry = call_and_log(MCTS_URL, "expand", payload, state, transport="http")
    return {
        "mcts_tree": res.get("mcts_tree"), 
        "current_selection": res.get("current_selection"), 
        "telemetry_logs": [log_entry], 
        "call_index": 1
    }

@orchestrator_node_metrics(node_name="semantic_agent")
def semantic_agent(state: OrchestratorState):
    logger.info("Executing Semantic Agent...")
    payload = {
        "config": state["config"],
        "task_description": state["perception_results"].get("task_description", ""),
        "current_tool": state["current_selection"].get("current_tool", ""),
        "all_error_analyses": state["mcts_tree"].get("all_error_analyses", []),
        "stage": state["current_selection"].get("stage", "evolve"),
        "user_input": state["user_input"],
        "data_prompt": state["perception_results"].get("data_prompt", "")
    }
    res, log_entry = call_and_log(SEMANTIC_URL, "retrieve_tutorials", payload, state, transport="http")
    
    semantic_results = {"tutorial_prompt": res.get("tutorial_prompt")}
    return {"semantic_results": semantic_results, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="coder_agent")
def coder_agent(state: OrchestratorState):
    logger.info("Executing Coder Agent...")
    parent_ctx = state["current_selection"].get("parent_context", {})
    
    payload = {
        "config": state["config"],
        "task_description": state["perception_results"].get("task_description", ""),
        "data_prompt": state["perception_results"].get("data_prompt", ""),
        "user_input": state["user_input"],
        "current_tool": state["current_selection"].get("current_tool", ""),
        "tool_prompt": state["perception_results"].get("tool_prompt", ""),
        "tutorial_prompt": state["semantic_results"].get("tutorial_prompt", ""),
        "all_error_analyses": state["mcts_tree"].get("all_error_analyses", []),
        "previous_python_code": parent_ctx.get("parent_code", ""),
        "previous_bash_script": parent_ctx.get("parent_bash", ""),
        "stage": state["current_selection"].get("stage", "evolve"),
        "iteration": state["mcts_tree"].get("iteration", 0),
        "node_id": state["current_selection"].get("node_id")
    }
    res, log_entry = call_and_log(CODER_URL, "generate_and_run", payload, state, transport="http")
    
    coding_results = {
        "python_code": res.get("python_code"),
        "bash_script": res.get("bash_script"),
        "stdout": res.get("stdout"),
        "stderr": res.get("stderr"),
        "decision": res.get("decision"),
        "validation_score": res.get("validation_score"),
        "error_message": res.get("error_message"),
        "error_analysis": res.get("error_summary")
    }
    return {"coding_results": coding_results, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="update_node")
def update_node(state: OrchestratorState):
    logger.info("MCTS: Update Node...")
    payload = {
        "action": "update",
        "mcts_tree": state["mcts_tree"],
        "current_selection": state["current_selection"],
        "coding_results": state["coding_results"]
    }
    res, log_entry = call_and_log(MCTS_URL, "update", payload, state, transport="http")
    return {
        "mcts_tree": res.get("mcts_tree"), 
        "current_selection": res.get("current_selection"), 
        "telemetry_logs": [log_entry], 
        "call_index": 1
    }

@orchestrator_node_metrics(node_name="backpropagate")
def backpropagate(state: OrchestratorState):
    logger.info("MCTS: Backpropagate...")
    payload = {
        "action": "backpropagate",
        "mcts_tree": state["mcts_tree"],
        "current_selection": state["current_selection"]
    }
    res, log_entry = call_and_log(MCTS_URL, "backpropagate", payload, state, transport="http")
    return {"mcts_tree": res, "telemetry_logs": [log_entry], "call_index": 1}

@orchestrator_node_metrics(node_name="finalize_results")
def finalize_results(state: OrchestratorState):
    logger.info("Finalizing MCTS search results...")
    payload = {
        "action": "finalize",
        "mcts_tree": state["mcts_tree"]
    }
    res, log_entry = call_and_log(MCTS_URL, "finalize", payload, state, transport="http")
    
    final_outcome = {
        "status": res.get("status"),
        "tree_visualization": res.get("tree_visualization")
    }
    return {"final_outcome": final_outcome, "telemetry_logs": [log_entry], "call_index": 1}


# ---------------------------------------------------------
# 4. Graph Edges & Routing
# ---------------------------------------------------------
def determine_start_route(state: OrchestratorState):
    """Route depending on S3 URI presence."""
    if state.get("s3_uri"):
        return "sync_s3_to_sandbox"
    return "perception_agent"

def route_after_select(state: OrchestratorState):
    """Translates the 'RouteAfterSelect' Choice state from Step Functions."""
    is_complete = state["current_selection"].get("is_complete", False)
    iteration_count = state["mcts_tree"].get("iteration", 0)
    
    if is_complete or iteration_count >= state["max_iterations"]:
        logger.info("MCTS selection marked search complete or max iterations reached.")
        return "finalize_results"
    return "expand_node"

# ---------------------------------------------------------
# 5. Build Graph
# ---------------------------------------------------------
def build_orchestrator_graph():
    workflow = StateGraph(OrchestratorState)
    
    # Add Nodes
    workflow.add_node("sync_s3_to_sandbox", sync_s3_to_sandbox)
    workflow.add_node("perception_agent", perception_agent)
    workflow.add_node("init_mcts", init_mcts)
    workflow.add_node("select_node", select_node)
    workflow.add_node("expand_node", expand_node)
    workflow.add_node("semantic_agent", semantic_agent)
    workflow.add_node("coder_agent", coder_agent)
    workflow.add_node("update_node", update_node)
    workflow.add_node("backpropagate", backpropagate)
    workflow.add_node("finalize_results", finalize_results)
    
    # Add Edges
    workflow.add_conditional_edges(START, determine_start_route)
    workflow.add_edge("sync_s3_to_sandbox", "perception_agent")
    workflow.add_edge("perception_agent", "init_mcts")
    workflow.add_edge("init_mcts", "select_node")
    
    workflow.add_conditional_edges(
        "select_node", 
        route_after_select, 
        {"finalize_results": "finalize_results", "expand_node": "expand_node"}
    )
    
    workflow.add_edge("expand_node", "semantic_agent")
    workflow.add_edge("semantic_agent", "coder_agent")
    workflow.add_edge("coder_agent", "update_node")
    workflow.add_edge("update_node", "backpropagate")
    workflow.add_edge("backpropagate", "select_node") # Loop back
    workflow.add_edge("finalize_results", END)
    
    return workflow.compile()


# ---------------------------------------------------------
# 6. Main Execution Wrapper
# ---------------------------------------------------------
def run_orchestration(
    input_data_folder: str,
    user_input: str,
    config: dict,
    max_iterations: int,
    s3_bucket: str = None,
    s3_uri: str = None,
    session_id: str = None,
    context_id: str = None,
    tracing: dict = None,
    **kwargs
) -> dict:
    # ── Metric logger setup ──
    metric_logger.setLevel(logging.INFO)
    if not metric_logger.handlers:
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter('%(message)s'))
        metric_logger.addHandler(_handler)
        
        try:
            log_file = Path(__file__).resolve().parent.parent / "metrics.jsonl"
            _file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            _file_handler.setFormatter(logging.Formatter('%(message)s'))
            metric_logger.addHandler(_file_handler)
        except Exception:
            try:
                _file_handler = logging.FileHandler("/tmp/metrics.jsonl", mode="a", encoding="utf-8")
                _file_handler.setFormatter(logging.Formatter('%(message)s'))
                metric_logger.addHandler(_file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file handler for metrics: {e}")

    # ── Initialize tracing context ──
    tracing_payload = {
        "session_id": session_id or config.get("session_id") or "unknown",
        "context_id": context_id or config.get("context_id") or uuid.uuid4().hex,
        "tracing": tracing or config.get("tracing", {}),
    }
    ctx.init_from_payload(tracing_payload)

    start_time_iso = datetime.utcnow().isoformat()
    process = psutil.Process()
    initial_mem = process.memory_info().rss
    start_time = time.time()
    
    initial_state = {
        "s3_uri": s3_uri,
        "input_data_folder": input_data_folder,
        "user_input": user_input,
        "config": config,
        "max_iterations": max_iterations,
        "s3_bucket": s3_bucket,
        "telemetry_logs": [],
        "call_index": 0
    }
    
    graph = build_orchestrator_graph()
    
    status = "SUCCESS"
    error_msg = None
    final_state = {}

    try:
        # Execute the LangGraph workflow
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Orchestration failed with unexpected exception")
        status = "FAILED"
        error_msg = str(e)
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
        })
        
    end_time_iso = datetime.utcnow().isoformat()
    total_duration = time.time() - start_time
    peak_mem = max(initial_mem, process.memory_info().rss)
    
    mcts_tree = final_state.get("mcts_tree", {})
    final_outcome = final_state.get("final_outcome", {})
    telemetry_logs = final_state.get("telemetry_logs", initial_state["telemetry_logs"])

    # Emit graph metrics
    step_count = final_state.get("call_index", 0)
    iteration_count = 0
    if isinstance(mcts_tree, dict):
        iteration_count = mcts_tree.get("iteration", 0)
        
    emit_event(metric_logger, {
        **ctx.snapshot(),
        "event_type": "psutil_metrics_graph",
        "graph_name": "mlorchestrator",
        "graph_e2e_s": round(total_duration, 4),
        "peak_RAM_GB": round(peak_mem / (1024**3), 4),
        "step_count": step_count,
        "iteration_count": iteration_count,
    })

    report = {
        "status": status,
        "error": error_msg,
        "orchestrator_start_time": start_time_iso,
        "orchestrator_end_time": end_time_iso,
        "total_duration_seconds": round(total_duration, 3),
        "input_parameters": {
            "s3_uri": s3_uri,
            "input_data_folder": input_data_folder,
            "user_input": user_input,
            "max_iterations": max_iterations,
            "config": config
        },
        "telemetry_logs": telemetry_logs,
        "final_outcome": final_outcome if final_outcome else None,
        "mcts_tree": mcts_tree if mcts_tree else None
    }
    
    # Publish terminal result to SQS
    publish_status(
        session_id=ctx.session_id.get(),
        node_name="orchestration",
        status="COMPLETED" if status == "SUCCESS" else "FAILED",
        duration_s=total_duration,
        error=error_msg,
        extra={"final_outcome": final_outcome if final_outcome else None}
    )

    return report
