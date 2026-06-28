import os
import sys
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import boto3

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from tree_store import TreeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcts_handler.app")

app = BedrockAgentCoreApp()

_s3_client_cache = None

def _get_s3_client():
    global _s3_client_cache
    if _s3_client_cache is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        _s3_client_cache = boto3.client("s3", region_name=region)
    return _s3_client_cache

def _get_tree(payload: dict):
    tree_input = payload.get("mcts_tree")
    if not tree_input:
        if "nodes" in payload or "s3_uri" in payload:
            tree_input = payload
        else:
            raise ValueError("Missing required parameter: 'mcts_tree'")

    s3_uri = None
    if isinstance(tree_input, str):
        if tree_input.startswith("s3://"):
            s3_uri = tree_input
    elif isinstance(tree_input, dict):
        s3_uri = tree_input.get("s3_uri")

    if s3_uri:
        logger.info(f"Loading MCTS tree from S3: {s3_uri}")
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        s3 = _get_s3_client()
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        full_tree = json.loads(content)
        return TreeStore.normalize_tree(full_tree)
    else:
        if not isinstance(tree_input, dict):
            raise ValueError(f"Invalid mcts_tree format: {tree_input}")
        return TreeStore.normalize_tree(tree_input)

def _save_tree(tree):
    s3_uri = tree.get("s3_uri")
    if s3_uri:
        logger.info(f"Saving MCTS tree to S3: {s3_uri}")
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(tree, indent=2),
            ContentType="application/json"
        )

def _make_lightweight_tree(tree):
    if "s3_uri" in tree:
        return {
            "s3_uri": tree["s3_uri"],
            "iteration": tree.get("iteration", 0),
            "max_iterations": tree.get("max_iterations", 10),
            "all_error_analyses": tree.get("all_error_analyses", []),
            "best_score": tree.get("best_score"),
            "best_code": tree.get("best_code", ""),
            "best_node_id": tree.get("best_node_id"),
            "best_validation_score": tree.get("best_validation_score"),
            "worst_validation_score": tree.get("worst_validation_score")
        }
    return tree

@app.entrypoint
def handle(payload: dict) -> dict:
    """
    AgentCore entrypoint for MCTS Handler.
    Inspects the incoming payload for an 'action' parameter and routes accordingly.
    """
    action = payload.get("action")
    if not action:
        # Fallback to init
        if "perception_results" in payload or "selected_tools" in payload:
            action = "init"
        else:
            raise ValueError("Missing required parameter: 'action'")

    logger.info(f"MCTS Handler Invoked with Action: {action}")

    try:
        if action == "init":
            perception_results = payload.get("perception_results", {})
            selected_tools = perception_results.get("selected_tools") or payload.get("selected_tools")
            if not selected_tools:
                raise ValueError("No tools selected or available in perception_results or payload.")
            mcts_config = payload.get("config", {}).get("mcts", {})
            
            tree = TreeStore.initialize(mcts_config, selected_tools)
            tree["iteration"] = 0
            tree["max_iterations"] = payload.get("max_iterations") or mcts_config.get("max_iterations", 10)
            tree["all_error_analyses"] = []
            tree["best_score"] = None
            tree["best_code"] = ""
            tree["best_node_id"] = None
            
            s3_bucket = payload.get("s3_bucket") or payload.get("config", {}).get("s3_bucket") or os.environ.get("S3_BUCKET_NAME")
            if s3_bucket:
                run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                s3_uri = f"s3://{s3_bucket}/mlorchestrator/runs/{run_id}/mcts_tree.json"
                tree["s3_uri"] = s3_uri
                _save_tree(tree)
                return _make_lightweight_tree(tree)
            
            return tree

        elif action == "select":
            tree = _get_tree(payload)
            node_id = TreeStore.select_node(tree)
            if node_id is None:
                logger.info("No expandable nodes. Finalizing.")
                return {
                    "node_id": None,
                    "stage": "root",
                    "depth": 0,
                    "is_complete": True
                }
                
            node = tree["nodes"][node_id]
            logger.info(f"Selected node {node_id} (stage={node.get('stage')}, depth={node.get('depth')})")
            return {
                "node_id": node_id,
                "stage": node.get("stage", "root"),
                "depth": node.get("depth", 0),
                "is_complete": False,
            }

        elif action == "expand":
            tree = _get_tree(payload)
            selection = payload.get("current_selection", {})
            if not selection:
                selection = {
                    "node_id": payload.get("node_id"),
                    "stage": payload.get("stage", "evolve"),
                    "depth": payload.get("depth", 0),
                    "is_complete": payload.get("is_complete", False)
                }
                
            parent_id = selection.get("node_id")
            if parent_id is not None:
                parent_id = int(parent_id)
            
            if selection.get("is_complete") or parent_id is None:
                return {"mcts_tree": _make_lightweight_tree(tree), "current_selection": selection}
                
            new_id = TreeStore.expand_node(tree, parent_id)
            child = tree["nodes"][new_id]
            logger.info(f"Created child node {new_id} (stage={child['stage']}, tool={child['tool_used']})")
            
            parent_context = TreeStore.get_parent_context(tree, new_id)
            selection.update({
                "node_id": new_id,
                "stage": child["stage"],
                "depth": child["depth"],
                "current_tool": child["tool_used"],
                "parent_context": parent_context
            })
            _save_tree(tree)
            return {
                "mcts_tree": _make_lightweight_tree(tree),
                "current_selection": selection
            }

        elif action == "update":
            tree = _get_tree(payload)
            coding_results = payload.get("coding_results", {})
            selection = payload.get("current_selection", {})
            
            if not coding_results:
                coding_results = {
                    "python_code": payload.get("python_code", ""),
                    "bash_script": payload.get("bash_script", ""),
                    "stdout": payload.get("stdout", ""),
                    "stderr": payload.get("stderr", ""),
                    "decision": payload.get("decision", "FIX"),
                    "validation_score": payload.get("validation_score"),
                    "error_analysis": payload.get("error_analysis", ""),
                    "error_message": payload.get("error_message", ""),
                }
            if not selection:
                selection = {
                    "node_id": payload.get("node_id"),
                    "stage": payload.get("stage"),
                    "depth": payload.get("depth", 0),
                    "is_complete": payload.get("is_complete", False)
                }
                
            node_id = selection.get("node_id")
            if node_id is not None:
                node_id = int(node_id)
            
            if node_id is None:
                return {"mcts_tree": _make_lightweight_tree(tree)}
                
            TreeStore.update_node(tree, node_id, coding_results)
            
            all_analyses = tree.get("all_error_analyses", [])
            node = tree["nodes"].get(node_id, {})
            if not node.get("is_successful") and coding_results.get("error_analysis"):
                tool = node.get("tool_used", "")
                analysis_str = str(coding_results.get("error_analysis"))[:1000]
                all_analyses.append(f"[Node {node_id} ({tool})] {analysis_str}")
                
            tree["all_error_analyses"] = all_analyses[-20:]
            tree["best_score"] = tree.get("best_validation_score")
            best_node_id = tree.get("best_node_id")
            if best_node_id is not None and best_node_id in tree["nodes"]:
                tree["best_code"] = tree["nodes"][best_node_id].get("python_code", "")
            
            _save_tree(tree)
            return {
                "mcts_tree": _make_lightweight_tree(tree),
                "current_selection": selection
            }

        elif action == "backpropagate":
            tree = _get_tree(payload)
            selection = payload.get("current_selection", {})
            if not selection:
                selection = {
                    "node_id": payload.get("node_id"),
                    "stage": payload.get("stage"),
                    "depth": payload.get("depth", 0),
                    "is_complete": payload.get("is_complete", False)
                }
                
            node_id = selection.get("node_id")
            if node_id is not None:
                node_id = int(node_id)
                TreeStore.backpropagate(tree, node_id)
                
            tree["iteration"] = tree.get("iteration", 0) + 1
            _save_tree(tree)
            return _make_lightweight_tree(tree)

        elif action == "finalize":
            tree = _get_tree(payload)
            tree_viz = TreeStore.visualize_tree(tree)
            logger.info(f"Final Tree Visualization:\n{tree_viz}")
            status = TreeStore.get_status(tree)
            logger.info(f"Tree status: {status}")
            
            status["best_score"] = tree.get("best_validation_score")
            best_node_id = tree.get("best_node_id")
            if best_node_id is not None and best_node_id in tree["nodes"]:
                status["best_code"] = tree["nodes"][best_node_id].get("python_code", "")
                status["best_node_id"] = best_node_id
            else:
                status["best_code"] = ""
                status["best_node_id"] = None
            
            return {"status": status, "tree_visualization": tree_viz}

        else:
            raise ValueError(f"Unknown action: '{action}'")

    except Exception as exc:
        logger.error(f"MCTS Handler execution error: {exc}", exc_info=True)
        return {
            "status": "FAILED",
            "error": str(exc),
        }

if __name__ == "__main__":
    app.run()
