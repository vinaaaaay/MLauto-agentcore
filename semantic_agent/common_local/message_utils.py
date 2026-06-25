"""
Serialize / deserialize LangChain messages for JSON transport
between orchestrator ↔ sub-agents.
"""

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)


def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    result = []
    for msg in messages:
        entry = {"type": msg.type, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name") and msg.name:
            entry["name"] = msg.name
        result.append(entry)
    return result


def deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    messages = []
    for item in data:
        t = item.get("type", "ai")
        c = item.get("content", "")
        if t == "human":
            messages.append(HumanMessage(content=c))
        elif t == "ai":
            kwargs = {}
            if item.get("tool_calls"):
                kwargs["tool_calls"] = item["tool_calls"]
            messages.append(AIMessage(content=c, **kwargs))
        elif t == "system":
            messages.append(SystemMessage(content=c))
        elif t == "tool":
            messages.append(ToolMessage(
                content=c,
                tool_call_id=item.get("tool_call_id", ""),
                name=item.get("name"),
            ))
        else:
            messages.append(AIMessage(content=c))
    return messages


def prepare_messages_for_summarization(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Ensure the message history is safe for an LLM call that doesn't expect tool metadata.
    Specifically:
    1. Strips 'tool_calls' from AIMessages.
    2. Converts ToolMessages to HumanMessages with an 'Observation:' prefix.
    """
    sanitized = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            # Strip tool_calls to avoid OpenAI API Error 400
            sanitized.append(AIMessage(content=msg.content))
        elif isinstance(msg, ToolMessage):
            # Convert to HumanMessage so the model sees it as plain text observation
            sanitized.append(HumanMessage(content=f"Observation: {msg.content}"))
        else:
            sanitized.append(msg)
    return sanitized