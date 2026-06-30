from langgraph.graph import StateGraph, START, END

from state import MLorchestratorState
from nodes import (
    sync_s3_to_sandbox,
    call_perception_agent,
    init_mcts,
    select_node,
    expand_node,
    call_memory_agent,
    call_coding_agent,
    update_node,
    backpropagate,
    finalize_results
)

def _route_after_select(state: MLorchestratorState) -> str:
    """Route after select_node: if complete, go to finalize_results; otherwise expand."""
    if state.get("is_complete"):
        return "finalize_results"
    return "expand_node"

def _should_continue(state: MLorchestratorState) -> str:
    """Decide whether to continue the MCTS search or stop/finalize."""
    if state.get("is_complete"):
        return "finalize_results"

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    if iteration >= max_iterations:
        return "finalize_results"

    return "select_node"

def build_orchestrator_graph():
    """
    Build and compile the unified LangGraph for MLorchestrator.
    
    Flow:
      START -> sync_s3_to_sandbox -> call_perception_agent -> init_mcts -> select_node
      select_node --(is_complete)--> finalize_results -> END
      select_node --(not complete)--> expand_node -> call_memory_agent -> call_coding_agent
      -> update_node -> backpropagate -> _should_continue
      _should_continue --(loop)--> select_node
      _should_continue --(done)--> finalize_results -> END
    """
    graph = StateGraph(MLorchestratorState)

    # 1. Add all Nodes
    graph.add_node("sync_s3_to_sandbox", sync_s3_to_sandbox)
    graph.add_node("call_perception_agent", call_perception_agent)
    graph.add_node("init_mcts", init_mcts)
    graph.add_node("select_node", select_node)
    graph.add_node("expand_node", expand_node)
    graph.add_node("call_memory_agent", call_memory_agent)
    graph.add_node("call_coding_agent", call_coding_agent)
    graph.add_node("update_node", update_node)
    graph.add_node("backpropagate", backpropagate)
    graph.add_node("finalize_results", finalize_results)

    # 2. Wire static/direct edges
    graph.add_edge(START, "sync_s3_to_sandbox")
    graph.add_edge("sync_s3_to_sandbox", "call_perception_agent")
    graph.add_edge("call_perception_agent", "init_mcts")
    graph.add_edge("init_mcts", "select_node")
    
    graph.add_edge("expand_node", "call_memory_agent")
    graph.add_edge("call_memory_agent", "call_coding_agent")
    graph.add_edge("call_coding_agent", "update_node")
    graph.add_edge("update_node", "backpropagate")
    
    graph.add_edge("finalize_results", END)

    # 3. Wire conditional/routing edges
    graph.add_conditional_edges("select_node", _route_after_select, {
        "finalize_results": "finalize_results",
        "expand_node": "expand_node"
    })

    graph.add_conditional_edges("backpropagate", _should_continue, {
        "select_node": "select_node",
        "finalize_results": "finalize_results"
    })

    return graph.compile()
