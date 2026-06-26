def __getattr__(name):
    if name == "build_coder_agent_graph":
        from .agent import build_coder_agent_graph
        return build_coder_agent_graph
    if name == "CoderAgentState":
        from .utils import CoderAgentState
        return CoderAgentState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
