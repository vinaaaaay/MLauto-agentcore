"""
Persistent MCTS Tree Store.

All operations are stateless pure functions that read/mutate a TreeState dict
and can be persisted to a single JSON file.  This is the core module that
decouples the MCTS tree structure from the in-memory orchestrator process,
enabling serverless or multi-process execution patterns.

Typical lifecycle (one iteration):
    tree = TreeStore.load(path)
    node_id = TreeStore.select_node(tree)
    new_id  = TreeStore.expand_node(tree, node_id)
    ctx     = TreeStore.get_parent_context(tree, new_id)
    # ... call coder agent with ctx ...
    TreeStore.update_node(tree, new_id, results)
    TreeStore.backpropagate(tree, new_id)
    TreeStore.save(path, tree)
"""

import json
import logging
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .node_state import (
    NodeState,
    TreeState,
    make_initial_tree_state,
    make_root_node,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence (load / save)
# ═══════════════════════════════════════════════════════════════════════════════

class TreeStore:
    """
    Static helper collection for persistent MCTS tree operations.
    Every public method is a classmethod/staticmethod — no instance state.
    """

    # ── Memory State Handlers ────────────────────────────────────────────

    @staticmethod
    def normalize_tree(raw: dict) -> TreeState:
        """
        Normalize a JSON-parsed TreeState dictionary.
        Step Functions JSON converts integer keys to strings. This converts them back.
        """
        if "nodes" in raw and isinstance(raw["nodes"], dict):
            nodes_dict = {}
            for k, v in raw["nodes"].items():
                nodes_dict[int(k)] = v
            raw["nodes"] = nodes_dict
        return raw  # type: ignore

    @staticmethod
    def initialize(
        mcts_config: Dict[str, Any],
        selected_tools: List[str],
    ) -> TreeState:
        """
        Create a fresh tree with only the root node.
        Returns the new TreeState.
        """
        tree = make_initial_tree_state(mcts_config, selected_tools)
        logger.info("Initialized new MCTS tree in memory")
        return tree

    # ── UCT Computation ──────────────────────────────────────────────────

    @staticmethod
    def _compute_uct(
        node: NodeState,
        parent_visits: int,
        exploration_constant: float,
        best_score: Optional[float],
        worst_score: Optional[float],
        failure_offset: float,
        failure_penalty_weight: float,
    ) -> float:
        """Compute UCT value for a single node (pure function)."""
        visits = node.get("visits", 0)
        if visits == 0:
            return float("inf")

        parent_v = max(1, parent_visits)

        # Failure penalty
        failure_visits = node.get("failure_visits", 0)
        normalized_fv = max(0, failure_visits - failure_offset)
        failure_penalty = -failure_penalty_weight * normalized_fv / visits

        # Validated reward contribution
        validated_visits = node.get("validated_visits", 0)
        if validated_visits > 0:
            validated_reward = node.get("validated_reward", 0.0)
            if (
                best_score is not None
                and worst_score is not None
                and best_score > worst_score
            ):
                avg_raw = validated_reward / validated_visits
                normalized = (avg_raw - worst_score) / (best_score - worst_score)
                weight = validated_visits / visits
                validated_contribution = weight * normalized
            else:
                validated_contribution = 1.0
        else:
            validated_contribution = 0.0

        exploitation = validated_contribution + failure_penalty
        exploration = exploration_constant * math.sqrt(math.log(parent_v) / visits)

        return exploitation + exploration

    # ── Selection ────────────────────────────────────────────────────────

    @staticmethod
    def select_node(tree: TreeState) -> Optional[int]:
        """
        Select the best node to expand using UCT.

        Returns the node_id of the selected node, or None if the tree
        is exhausted (no expandable leaves).
        """
        nodes = tree["nodes"]
        config = tree.get("mcts_config", {})

        root_id = -1
        root = nodes.get(root_id)
        if root is None:
            return None

        initial_root_children = config.get("initial_root_children", 3)

        # Phase 1: Force root diversification
        if len(root.get("child_ids", [])) < initial_root_children:
            return root_id

        # Phase 2: UCT selection among expandable leaves
        exploration_constant = config.get("exploration_constant", 1.414)
        max_debug_children = config.get("max_debug_children", 2)
        max_evolve_children = config.get("max_evolve_children", 2)
        failure_offset = config.get("failure_offset", 0)
        failure_penalty_weight = config.get("failure_penalty_weight", 0.5)
        best_score = tree.get("best_validation_score")
        worst_score = tree.get("worst_validation_score")

        expandable = TreeStore._get_expandable_leaves(
            nodes, root_id, max_debug_children, max_evolve_children
        )

        if not expandable:
            return None

        best_uct = -float("inf")
        best_id = None
        for nid in expandable:
            node = nodes[nid]
            parent_id = node.get("parent_id")
            parent_visits = nodes[parent_id].get("visits", 0) if parent_id is not None else 1

            uct = TreeStore._compute_uct(
                node,
                parent_visits,
                exploration_constant,
                best_score,
                worst_score,
                failure_offset,
                failure_penalty_weight,
            )
            if uct > best_uct:
                best_uct = uct
                best_id = nid

        return best_id

    @staticmethod
    def _get_expandable_leaves(
        nodes: Dict[int, NodeState],
        root_id: int,
        max_debug_children: int,
        max_evolve_children: int,
    ) -> List[int]:
        """Return IDs of all non-terminal nodes that can still be expanded."""
        result = []
        for nid, node in nodes.items():
            if nid == root_id:
                continue
            if node.get("is_terminal", False):
                continue

            child_ids = node.get("child_ids", [])
            num_children = len(child_ids)
            is_leaf = num_children == 0

            if is_leaf:
                result.append(nid)
            elif node.get("is_successful", False) and num_children < max_evolve_children:
                result.append(nid)
            elif not node.get("is_successful", False) and num_children < max_debug_children:
                result.append(nid)

        return result

    # ── Expansion ────────────────────────────────────────────────────────

    @staticmethod
    def expand_node(tree: TreeState, parent_id: int) -> int:
        """
        Create a new child node under ``parent_id``.

        Automatically decides evolve vs. debug based on parent success.
        Returns the new node's ID.
        """
        nodes = tree["nodes"]
        parent = nodes[parent_id]
        selected_tools = tree.get("selected_tools", [])

        # Allocate new time step
        new_id = tree["next_time_step"]
        tree["next_time_step"] = new_id + 1

        # Decide stage and tool
        if parent.get("stage") == "root" or parent.get("is_successful", False):
            stage = "evolve"
            tool = TreeStore._get_next_tool(tree)
        else:
            stage = "debug"
            tool = parent.get("tool_used", "")

        child = NodeState(
            node_id=new_id,
            parent_id=parent_id,
            child_ids=[],
            ctime=time.time(),
            depth=parent.get("depth", 0) + 1,
            stage=stage,
            visits=0,
            validated_visits=0,
            failure_visits=0,
            unvalidated_visits=0,
            validated_reward=0.0,
            is_successful=False,
            is_debug_successful=False,
            is_terminal=False,
            debug_attempts=0,
            tool_used=tool,
            tools_available=selected_tools,
            python_code="",
            bash_script="",
            stdout="",
            stderr="",
            execution_time=0.0,
            processing_time=0.0,
            ai_call_time=0.0,
            error_message="",
            error_analysis="",
            validation_score=None,
            expected_child_count=0,
        )

        # Register in tree
        nodes[new_id] = child
        parent["child_ids"].append(new_id)

        logger.info(
            f"Expanded: new node {new_id} (stage={stage}, tool={tool}) "
            f"under parent {parent_id}"
        )
        return new_id

    @staticmethod
    def _get_next_tool(tree: TreeState) -> str:
        """Rotate through available tools (round-robin)."""
        tools = tree.get("selected_tools", [])
        if not tools:
            return "machine learning"
        idx = tree.get("tool_index", 0)
        tool = tools[idx % len(tools)]
        tree["tool_index"] = idx + 1
        return tool

    # ── Node Update ──────────────────────────────────────────────────────

    @staticmethod
    def update_node(tree: TreeState, node_id: int, results: Dict[str, Any]) -> None:
        """
        Write execution results into a node.

        ``results`` should contain keys like:
            python_code, bash_script, stdout, stderr,
            validation_score, decision, error_message, error_analysis,
            execution_time, processing_time, ai_call_time
        """
        node = tree["nodes"][node_id]

        node["python_code"] = results.get("python_code", "")
        node["bash_script"] = results.get("bash_script", "")
        node["stdout"] = results.get("stdout", "")
        node["stderr"] = results.get("stderr", "")
        node["execution_time"] = results.get("execution_time", 0.0)
        node["processing_time"] = results.get("processing_time", 0.0)
        node["ai_call_time"] = results.get("ai_call_time", 0.0)
        node["error_message"] = results.get("error_message", "")
        node["error_analysis"] = results.get("error_analysis", "")

        validation_score = results.get("validation_score")
        if validation_score is not None:
            try:
                validation_score = float(validation_score)
            except (ValueError, TypeError):
                validation_score = None
        node["validation_score"] = validation_score

        decision = results.get("decision", "FIX")
        node["is_successful"] = (decision == "SUCCESS")

        # Update global best/worst scores
        if validation_score is not None:
            best = tree.get("best_validation_score")
            if best is None or validation_score > best:
                tree["best_validation_score"] = validation_score
                tree["best_node_id"] = node_id
                logger.info(
                    f"*** NEW BEST score: {validation_score:.4f} on node {node_id} ***"
                )

            worst = tree.get("worst_validation_score")
            if worst is None or validation_score < worst:
                tree["worst_validation_score"] = validation_score

        logger.info(
            f"Updated node {node_id}: successful={node['is_successful']}, "
            f"score={node['validation_score']}"
        )

    # ── Backpropagation ──────────────────────────────────────────────────

    @staticmethod
    def backpropagate(tree: TreeState, node_id: int) -> None:
        """
        Walk the parent chain from ``node_id`` upward, updating visit
        counts and rewards. Also handles debug promotion and terminal
        marking.
        """
        nodes = tree["nodes"]
        config = tree.get("mcts_config", {})
        max_debug_depth = config.get("max_debug_depth", 3)

        node = nodes[node_id]
        decision = "SUCCESS" if node.get("is_successful", False) else "FIX"
        validation_score = node.get("validation_score")
        is_failure = decision != "SUCCESS"
        is_validated = validation_score is not None

        # Compute processing time
        ctime = node.get("ctime", 0)
        if ctime:
            node["processing_time"] = time.time() - ctime

        # ── Debug Promotion ──────────────────────────────────────────
        if decision == "SUCCESS" and node.get("stage") == "debug":
            origin_id = TreeStore._find_debug_origin(nodes, node_id)
            if origin_id is not None:
                origin = nodes[origin_id]
                old_parent_id = node["parent_id"]
                new_parent_id = origin.get("parent_id")

                if new_parent_id is not None and old_parent_id is not None:
                    # Detach from old parent
                    old_parent = nodes[old_parent_id]
                    if node_id in old_parent["child_ids"]:
                        old_parent["child_ids"].remove(node_id)

                    # Re-attach to grandparent
                    new_parent = nodes[new_parent_id]
                    new_parent["child_ids"].append(node_id)
                    node["parent_id"] = new_parent_id

                    # Mark the failed origin branch as terminal
                    TreeStore._mark_terminal(nodes, origin_id)

                    node["is_debug_successful"] = True
                    logger.info(
                        f"Promoted debug node {node_id}, "
                        f"marked origin {origin_id} terminal"
                    )

        # ── Failed Debug: increment parent debug_attempts ────────────
        if is_failure and node.get("stage") == "debug":
            parent_id = node.get("parent_id")
            if parent_id is not None and parent_id in nodes:
                parent = nodes[parent_id]
                parent["debug_attempts"] = parent.get("debug_attempts", 0) + 1
                if parent["debug_attempts"] >= max_debug_depth:
                    TreeStore._mark_terminal(nodes, parent_id)
                    logger.warning(
                        f"Parent {parent_id} reached max debug depth "
                        f"({max_debug_depth}), marked terminal"
                    )

        # ── Walk parent chain updating stats ─────────────────────────
        current_id: Optional[int] = node_id
        while current_id is not None:
            cur = nodes[current_id]
            cur["visits"] = cur.get("visits", 0) + 1
            if is_failure:
                cur["failure_visits"] = cur.get("failure_visits", 0) + 1
            elif is_validated and validation_score is not None:
                cur["validated_visits"] = cur.get("validated_visits", 0) + 1
                cur["validated_reward"] = cur.get("validated_reward", 0.0) + validation_score
            else:
                cur["unvalidated_visits"] = cur.get("unvalidated_visits", 0) + 1
            current_id = cur.get("parent_id")

        logger.info(f"Backpropagated from node {node_id}")

    @staticmethod
    def _find_debug_origin(
        nodes: Dict[int, NodeState], node_id: int
    ) -> Optional[int]:
        """
        Trace up the parent chain to find the original non-debug node
        that started a debug sequence. Returns None if not found or
        if the origin is the root.
        """
        current_id = nodes[node_id].get("parent_id")
        while current_id is not None:
            current = nodes[current_id]
            if current.get("stage") != "debug":
                # Found the origin — but skip if it's the root
                if current_id == -1:
                    return None
                return current_id
            current_id = current.get("parent_id")
        return None

    @staticmethod
    def _mark_terminal(nodes: Dict[int, NodeState], node_id: int) -> None:
        """Recursively mark a node and all its descendants as terminal."""
        node = nodes.get(node_id)
        if node is None:
            return
        node["is_terminal"] = True
        for child_id in node.get("child_ids", []):
            TreeStore._mark_terminal(nodes, child_id)

    # ── Context Retrieval ────────────────────────────────────────────────

    @staticmethod
    def get_parent_context(tree: TreeState, node_id: int) -> Dict[str, str]:
        """
        Return the parent node's code and error context for the coder agent.
        Returns empty strings if the parent is the root or missing.
        """
        nodes = tree["nodes"]
        node = nodes.get(node_id)
        if node is None:
            return {"parent_code": "", "parent_bash": "", "parent_error": ""}

        parent_id = node.get("parent_id")
        if parent_id is None or parent_id not in nodes:
            return {"parent_code": "", "parent_bash": "", "parent_error": ""}

        parent = nodes[parent_id]
        if parent.get("stage") == "root":
            return {"parent_code": "", "parent_bash": "", "parent_error": ""}

        return {
            "parent_code": parent.get("python_code", ""),
            "parent_bash": parent.get("bash_script", ""),
            "parent_error": parent.get("error_message", ""),
        }

    # ── Visualization ────────────────────────────────────────────────────

    @staticmethod
    def visualize_tree(tree: TreeState) -> str:
        """Generate ASCII tree visualization from persistent state."""
        nodes = tree["nodes"]
        lines: List[str] = []
        TreeStore._visualize_subtree(nodes, -1, "", True, lines)
        return "\n".join(lines)

    @staticmethod
    def _visualize_subtree(
        nodes: Dict[int, NodeState],
        node_id: int,
        prefix: str,
        is_last: bool,
        lines: List[str],
    ) -> None:
        node = nodes.get(node_id)
        if node is None:
            return

        connector = "└── " if is_last else "├── "

        if node_id == -1:
            label = "Root"
        else:
            status = "✓" if node.get("is_successful") else "✗"
            score = node.get("validation_score")
            score_str = f"score={score:.4f}" if score is not None else "no score"
            terminal = " [TERMINAL]" if node.get("is_terminal") else ""
            label = (
                f"Node {node_id} "
                f"[{node.get('stage')}|{node.get('tool_used', '')}|"
                f"{status}|v={node.get('visits', 0)}|{score_str}]{terminal}"
            )

        lines.append(f"{prefix}{connector}{label}")
        child_ids = node.get("child_ids", [])
        for i, cid in enumerate(child_ids):
            extension = "    " if is_last else "│   "
            TreeStore._visualize_subtree(
                nodes, cid, prefix + extension, i == len(child_ids) - 1, lines
            )

    # ── Summary / Status ─────────────────────────────────────────────────

    @staticmethod
    def get_status(tree: TreeState) -> Dict[str, Any]:
        """Return a concise summary of the current tree state."""
        nodes = tree["nodes"]
        total = len(nodes) - 1  # Exclude root
        successful = sum(
            1 for n in nodes.values()
            if n.get("is_successful") and n.get("node_id") != -1
        )
        terminal = sum(
            1 for n in nodes.values()
            if n.get("is_terminal") and n.get("node_id") != -1
        )
        config = tree.get("mcts_config", {})
        expandable = TreeStore._get_expandable_leaves(
            nodes, -1,
            config.get("max_debug_children", 2),
            config.get("max_evolve_children", 2),
        )

        return {
            "total_nodes": total,
            "successful_nodes": successful,
            "terminal_nodes": terminal,
            "expandable_nodes": len(expandable),
            "best_node_id": tree.get("best_node_id"),
            "best_validation_score": tree.get("best_validation_score"),
            "worst_validation_score": tree.get("worst_validation_score"),
        }
