def __getattr__(name):
    if name == "build_semantic_agent_graph":
        from .agent import build_semantic_agent_graph
        return build_semantic_agent_graph
    if name == "SemanticAgentState":
        from .utils import SemanticAgentState
        return SemanticAgentState
    if name == "TutorialInfo":
        from .utils import TutorialInfo
        return TutorialInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
