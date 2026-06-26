"""
State definition for the Perception Agent LangGraph pipeline.

Fully self-contained — no imports from external FAME or MLauto modules.
"""

from typing import TypedDict, Optional, List, Dict, Any


class PerceptionAgentState(TypedDict, total=False):
    """
    State representing the data flow through the Perception Agent graph.

    Supports a flexible dictionary model for easy extension.
    Designed to be completely standalone — deployable as a black-box service.
    """

    # ── Configuration ──
    config: Dict[str, Any]

    # ── Inputs ──
    input_data_folder: str
    output_folder: str
    user_input: str

    # ── Perception Outputs ──
    data_prompt: str
    description_files: List[str]
    task_description: str
    selected_tools: List[str]
    current_tool: str
    tool_prompt: str

    # ── Error Context (passed through for downstream use) ──
    all_error_analyses: List[str]
