"""
Centralized metrics context — single source of truth for all correlation IDs.

Three-ID span model:
    trace_id        — root correlation, same across entire query execution tree
    span_id         — unique to THIS agent invocation
    parent_span_id  — span_id of whoever invoked this agent (None for root)
"""

import uuid
import contextvars


class MetricsContext:
    """
    Holds all context variables for metrics emission and trace propagation.
    One instance per agent module. ContextVars are coroutine/thread-safe.
    """

    def __init__(self, agent_id: str = "unknown"):
        self.agent_id       = agent_id
        self.session_id     = contextvars.ContextVar("session_id",     default="unknown")
        self.context_id     = contextvars.ContextVar("context_id",     default="unknown")
        self.trace_id       = self.context_id
        self.span_id        = contextvars.ContextVar("span_id",        default="unknown")
        self.parent_span_id = contextvars.ContextVar("parent_span_id", default=None)
        self.state_id       = contextvars.ContextVar("state_id",       default="unknown")
        self.query_id       = contextvars.ContextVar("query_id",       default="unknown")
        self.node_name      = contextvars.ContextVar("node_name",      default="unknown")
        self.custom_wait_time = contextvars.ContextVar("custom_wait_time", default=None)
        self.openai_processing_ms_ledger = {}

    def init_from_payload(self, payload: dict):
        """Called once when an agent receives a request."""
        tracing = payload.get("tracing", {})

        self.agent_id = (
            payload.get("agent_id")
            or tracing.get("agent_id")
            or self.agent_id
        )

        self.context_id.set(
            tracing.get("context_id")
            or payload.get("context_id")
            or tracing.get("trace_id")
            or payload.get("trace_id")
            or uuid.uuid4().hex
        )

        self.span_id.set(uuid.uuid4().hex[:24])
        self.parent_span_id.set(tracing.get("parent_span_id"))
        self.state_id.set(str(uuid.uuid4()))
        self.session_id.set(payload.get("session_id", "unknown"))
        self.query_id.set(payload.get("query_id", self.context_id.get()))

    def child_context(self) -> dict:
        """Build the tracing dict to inject into a sub-agent's payload."""
        return {
            "context_id": self.context_id.get(),
            "parent_span_id": self.span_id.get(),
        }

    def snapshot(self) -> dict:
        """Returns all correlation IDs for inclusion in every emitted event."""
        return {
            "session_id":     self.session_id.get(),
            "agent_id":       self.agent_id,
            "context_id":     self.context_id.get(),
            "span_id":        self.span_id.get(),
            "parent_span_id": self.parent_span_id.get(),
            "state_id":       self.state_id.get(),
            "query_id":       self.query_id.get(),
        }
