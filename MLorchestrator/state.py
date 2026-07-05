from typing import TypedDict, Optional, List, Dict, Any

class MLorchestratorState(TypedDict, total=False):
    # ── User Inputs ──
    input_data_folder: str
    output_folder: str
    user_input: str
    config: dict
    s3_uri: str
    s3_bucket: str

    # ── Perception Results (cached from A2A) ──
    data_prompt: str
    task_description: str
    selected_tools: List[str]
    current_tool: str
    tool_prompt: str

    # ── MCTS State ──
    mcts_tree: dict
    current_selection: dict
    iteration: int
    max_iterations: int
    is_complete: bool

    # ── Current Node Context ──
    node_id: int
    stage: str                   # "evolve" | "debug"
    depth: int

    # ── Memory Result (from A2A) ──
    tutorial_prompt: str
    semantic_results: Dict[str, Any]

    # ── Coding Result (from A2A) ──
    python_code: str
    bash_script: str
    stdout: str
    stderr: str
    decision: str                # "SUCCESS" | "FIX"
    error_summary: str
    validation_score: Optional[float]
    error_analysis: str
    error_message: str
    coding_results: Dict[str, Any]

    # ── Accumulated State ──
    all_error_analyses: List[str]
    best_score: Optional[float]
    best_code: str
    best_node_id: Optional[int]
    tree_visualization: str
