from typing import Any, Dict, List, Optional, TypedDict

class CoderAgentState(TypedDict, total=False):
    # ── Context Inputs ──
    task_description: str
    data_prompt: str
    user_input: str
    current_tool: str
    tool_prompt: str
    tutorial_prompt: str
    all_error_analyses: List[str]

    # ── Run Configuration ──
    config: Dict[str, Any]
    output_folder: str
    sandbox_client: Any

    # ── Current iteration tracking ──
    iteration: int
    node_id: int
    stage: str  # "root", "evolve", or "debug"

    # Previous attempts (if improving/debugging)
    previous_python_code: str
    previous_bash_script: str

    # ── Outputs ──
    python_code: str
    python_file_path: str
    bash_script: str
    stdout: str
    stderr: str
    decision: str  # "SUCCESS" or "FIX"
    error_summary: Optional[str]
    validation_score: Optional[float]
    error_message: str
