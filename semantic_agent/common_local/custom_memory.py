"""
Custom Memory Client — SHORT-TERM memory only.

Write path:
    create_event  →  stores raw conversation turns in a session
                     (same API as AgentCoreMemoryStore._handle_put)

Read path:
    list_events   →  retrieves raw session events back
                     (NOT retrieve_memory_records, which is long-term / billed separately)

AgentCore enforces that write payloads are BaseMessage objects.
This constraint is validated here, and conversion uses the same
helper the upstream AgentCoreMemoryStore relies on.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
# from langgraph_checkpoint_aws.agentcore.helpers import (
#     convert_langchain_messages_to_event_messages,
# )

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Args:
        memory_id:  AgentCore Memory resource ID.
        actor_id:   Identity of the user / actor.
        thread_id:  Session scope (maps to sessionId).
        **boto3_kwargs:  Passed to boto3.client().
    """

    def __init__(
        self,
        *,
        memory_id: str,
        actor_id: str,
        thread_id: str,
        **boto3_kwargs: Any,
    ):
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.thread_id = thread_id

        config = Config(
            user_agent_extra="x-client-framework:custom_memory_client",
            retries={"max_attempts": 4, "mode": "adaptive"},
            read_timeout=900,
            connect_timeout=60,
        )
        self._client = boto3.client(
            "bedrock-agentcore", config=config, **boto3_kwargs
        )

    # ── WRITE (short-term) ───────────────────────────────────

    def write(self, messages: list[BaseMessage]) -> bool:
        """
        Persist messages as a conversation event in the session.

        Validates every element is a BaseMessage (AgentCore constraint),
        then delegates to convert_langchain_messages_to_event_messages
        — the same path AgentCoreMemoryStore._handle_put uses.
        """
        for msg in messages:
            if not isinstance(msg, BaseMessage):
                raise ValueError(
                    f"AgentCore Memory only accepts BaseMessage objects. "
                    f"Got {type(msg).__name__}."
                )

        event_messages = self._convert_langchain_messages_to_event_messages(messages)
        if not event_messages:
            logger.warning("No valid event messages — nothing written.")
            return False

        payloads = [
            {"conversational": {"content": {"text": text}, "role": role}}
            for text, role in event_messages
        ]

        start = time.time()
        try:
            self._client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=self.thread_id,
                eventTimestamp=datetime.now(timezone.utc),
                payload=payloads,
            )
            logger.debug(
                "memory_write ok session=%s events=%d ms=%.1f",
                self.thread_id, len(payloads), self._ms(start),
            )
            return True
        except Exception as e:
            logger.error("memory_write error session=%s: %s", self.thread_id, e)
            return False

    # ── READ (short-term) ────────────────────────────────────

    def read(self) -> list[BaseMessage]:
        """
        Retrieve raw conversation events from the session.

        Uses list_events (short-term) — NOT retrieve_memory_records (long-term).
        Paginates automatically to collect full session history.

        NOTE: If your SDK version uses a different method name, run:
            for op in sorted(client.meta.service_model.operation_names):
                if 'event' in op.lower() or 'session' in op.lower():
                    print(op)
        and update the call below accordingly.
        """
        all_messages: list[BaseMessage] = []
        next_token = None
        start = time.time()

        try:
            while True:
                kwargs: dict[str, Any] = {
                    "memoryId": self.memory_id,
                    "sessionId": self.thread_id,
                    "actorId": self.actor_id,
                }
                if next_token:
                    kwargs["nextToken"] = next_token

                response = self._client.list_events(**kwargs)

                for event in response.get("events", []):
                    all_messages.extend(self._parse_event(event))

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.debug(
                "memory_read ok session=%s messages=%d ms=%.1f",
                self.thread_id, len(all_messages), self._ms(start),
            )
            return all_messages

        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ResourceNotFoundException", "SessionNotFoundException"):
                logger.debug("memory_read not_found session=%s", self.thread_id)
                return []
            logger.error("memory_read error session=%s: %s", self.thread_id, e)
            return []
        except Exception as e:
            logger.error("memory_read error session=%s: %s", self.thread_id, e)
            return []

    # ── internal parsing ─────────────────────────────────────
    
    @staticmethod
    def _convert_langchain_messages_to_event_messages(
        messages: list[BaseMessage]
    ) -> list[tuple[str, str]]:
        """
        Local version of the helper to avoid LangChain deprecation warnings.
        Converts LangChain messages to AgentCore event tuples (text, role).
        """
        converted = []
        for msg in messages:
            # Skip empty content
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not text.strip():
                continue

            # Map LangChain roles to Bedrock roles
            if msg.type == "human":
                role = "USER"
            elif msg.type == "ai":
                role = "ASSISTANT"
            elif msg.type == "tool":
                role = "TOOL"
            elif msg.type == "system":
                role = "ASSISTANT"  # System mapped to Assistant or handled as needed
            else:
                logger.warning(f"Skipping unsupported message type: {msg.type}")
                continue
            
            converted.append((text, role))
        return converted

    @staticmethod
    def _parse_event(event: dict) -> list[BaseMessage]:
        """
        Convert a single API event (which may contain multiple
        conversational payload items) into BaseMessage objects.

        Reverses the mapping that convert_langchain_messages_to_event_messages
        produces:
            USER       → HumanMessage
            ASSISTANT  → AIMessage
            TOOL       → ToolMessage
            *          → AIMessage (fallback)
        """
        messages: list[BaseMessage] = []
        for payload_item in event.get("payload", []):
            conv = payload_item.get("conversational", {})
            if not conv:
                continue
            role = conv.get("role", "ASSISTANT").upper()
            content = conv.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else str(content)

            if role == "USER":
                messages.append(HumanMessage(content=text))
            elif role == "ASSISTANT":
                messages.append(AIMessage(content=text))
            elif role == "TOOL":
                messages.append(ToolMessage(content=text, tool_call_id=""))
            else:
                messages.append(AIMessage(content=text))
        return messages

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.time() - start) * 1000, 2)