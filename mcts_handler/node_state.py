"""
Serializable node and tree state definitions for persistent MCTS.

Instead of in-memory Node objects with parent/child pointers,
NodeState uses integer IDs for relationships so the entire tree
can be serialized to/from JSON and shared across processes.
"""

import time
from typing import Any, Dict, List, Optional, TypedDict


class NodeState(TypedDict, total=False):
    """
    A single node in the MCTS tree, fully serializable to JSON.
    Mirrors every field of mcts.node.Node but replaces object
    references with integer IDs.
    """

    # Identity & structure
    node_id: int                     # Unique ID (== time_step in original Node)
    parent_id: Optional[int]         # ID of parent node (None for root)
    child_ids: List[int]             # IDs of child nodes

    # Creation time
    ctime: float

    # Position
    depth: int                       # Depth in tree (root=0)

    # Solution stage
    stage: str                       # "root" | "evolve" | "debug"

    # MCTS statistics
    visits: int
    validated_visits: int
    failure_visits: int
    unvalidated_visits: int
    validated_reward: float

    # Node state tracking
    is_successful: bool
    is_debug_successful: bool
    is_terminal: bool
    debug_attempts: int

    # Tool
    tool_used: str
    tools_available: List[str]

    # Solution artifacts
    python_code: str
    bash_script: str

    # Execution results
    stdout: str
    stderr: str
    execution_time: float
    processing_time: float
    ai_call_time: float
    error_message: str
    error_analysis: str

    # Evaluation
    validation_score: Optional[float]

    # Expansion control
    expected_child_count: int


class TreeState(TypedDict, total=False):
    """
    Complete persistent state of the MCTS tree.
    Loaded from / saved to a single JSON file.
    """

    # All nodes keyed by node_id (serialized as a list, loaded into a dict)
    nodes: Dict[int, NodeState]

    # Global counters
    next_time_step: int
    tool_index: int

    # Score tracking
    best_node_id: Optional[int]
    best_validation_score: Optional[float]
    worst_validation_score: Optional[float]

    # Config snapshot (MCTS params only)
    mcts_config: Dict[str, Any]

    # Available tools
    selected_tools: List[str]


def make_root_node() -> NodeState:
    """Create the virtual root node (node_id = -1)."""
    return NodeState(
        node_id=-1,
        parent_id=None,
        child_ids=[],
        ctime=time.time(),
        depth=0,
        stage="root",
        visits=0,
        validated_visits=0,
        failure_visits=0,
        unvalidated_visits=0,
        validated_reward=0.0,
        is_successful=True,  # Root is a virtual node, always "successful"
        is_debug_successful=False,
        is_terminal=False,
        debug_attempts=0,
        tool_used="",
        tools_available=[],
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


def make_initial_tree_state(
    mcts_config: Dict[str, Any],
    selected_tools: List[str],
) -> TreeState:
    """
    Create an initial TreeState with only the root node.
    This is the starting point for a new MCTS search.
    """
    root = make_root_node()
    return TreeState(
        nodes={root["node_id"]: root},
        next_time_step=0,
        tool_index=0,
        best_node_id=None,
        best_validation_score=None,
        worst_validation_score=None,
        mcts_config=mcts_config,
        selected_tools=selected_tools,
    )
