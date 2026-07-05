from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import logging
import json
import time
import httpx
from datetime import datetime, timezone
from typing import Any

from .metrics_context import MetricsContext

logger = logging.getLogger(__name__)


class OpenAIHeaderInterceptor(httpx.BaseTransport):
    """
    HTTPX interceptor that captures the 'openai-processing-ms' header
    and populates the provided MetricsContext.
    """

    def __init__(self, transport: httpx.BaseTransport, ctx: MetricsContext):
        self.transport = transport
        self.ctx = ctx

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.transport.handle_request(request)
        self._capture_header(response.headers)
        return response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.transport.handle_async_request(request)
        self._capture_header(response.headers)
        return response

    def _capture_header(self, headers: httpx.Headers):
        processing_ms = headers.get("openai-processing-ms")
        if processing_ms:
            span_id = self.ctx.span_id.get()
            if span_id:
                try:
                    self.ctx.openai_processing_ms_ledger[span_id] = float(processing_ms)
                except (ValueError, TypeError):
                    pass


class SessionMetricsCallback(BaseCallbackHandler):
    """
    Callback that emits an 'llm_call' event for every LLM invocation.
    """

    def __init__(self, ctx: MetricsContext, metric_logger: logging.Logger):
        self.ctx = ctx
        self.metric_logger = metric_logger
        self.llm_starts = {}
        self.tool_starts = {}

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id, parent_run_id, **kwargs):
        node_name = self.ctx.node_name.get()
        self.tool_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "tool_name": serialized.get("name") if serialized else "unknown",
            "input": input_str,
        }

    def on_tool_end(self, output: Any, *, run_id, parent_run_id, **kwargs):
        latency = 0
        node_name = "unknown"
        tool_name = "unknown"
        tool_input = ""
        if run_id in self.tool_starts:
            start_info = self.tool_starts.pop(run_id)
            latency = (time.time() - start_info["start"]) * 1000
            node_name = start_info["node_name"]
            tool_name = start_info["tool_name"]
            tool_input = start_info["input"]

        self.metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "tool_call",
            **self.ctx.snapshot(),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "node_name": node_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": str(output),
            "latency_ms": round(latency, 2),
            "status": "success",
        }))

    def on_tool_error(self, error: BaseException, *, run_id, parent_run_id, **kwargs):
        latency = 0
        node_name = "unknown"
        tool_name = "unknown"
        tool_input = ""
        if run_id in self.tool_starts:
            start_info = self.tool_starts.pop(run_id)
            latency = (time.time() - start_info["start"]) * 1000
            node_name = start_info["node_name"]
            tool_name = start_info["tool_name"]
            tool_input = start_info["input"]

        self.metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "tool_call",
            **self.ctx.snapshot(),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "node_name": node_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": str(error),
            "latency_ms": round(latency, 2),
            "status": "error",
        }))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs):
        node_name = self.ctx.node_name.get()
        input_bytes = sum(len(p.encode("utf-8")) for p in prompts) if prompts else 0

        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes,
        }

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id, **kwargs):
        node_name = self.ctx.node_name.get()

        input_bytes = 0
        if messages:
            for msg_list in messages:
                for msg in msg_list:
                    if hasattr(msg, "content") and msg.content:
                        if isinstance(msg.content, str):
                            input_bytes += len(msg.content.encode("utf-8"))
                        else:
                            input_bytes += len(json.dumps(msg.content).encode("utf-8"))

                    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                        input_bytes += len(json.dumps(msg.additional_kwargs).encode("utf-8"))
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        input_bytes += len(json.dumps(msg.tool_calls).encode("utf-8"))

        invocation_params = kwargs.get("invocation_params", {})
        if "tools" in invocation_params:
            input_bytes += len(json.dumps(invocation_params["tools"]).encode("utf-8"))
        elif "functions" in invocation_params:
            input_bytes += len(json.dumps(invocation_params["functions"]).encode("utf-8"))
        if "tools" in kwargs and "tools" not in invocation_params:
            input_bytes += len(json.dumps(kwargs["tools"]).encode("utf-8"))

        self.llm_starts[run_id] = {
            "start": time.time(),
            "node_name": node_name,
            "input_bytes": input_bytes,
        }

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id, **kwargs):
        latency = 0
        node_name = "unknown"
        input_bytes = 0
        if run_id in self.llm_starts:
            start_info = self.llm_starts.pop(run_id)
            latency = (time.time() - start_info["start"]) * 1000
            node_name = start_info["node_name"]
            input_bytes = start_info["input_bytes"]

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        output_bytes = 0

        # ── Extract token counts from response ──
        # Extract from llm_output if present
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cached_tokens = usage.get("cached_tokens") or usage.get("cache_read_input_tokens") or usage.get("cache_read") or 0
            
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens", 0) or details.get("reasoning", 0)
                
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                cached_tokens = cached_tokens or prompt_details.get("cached_tokens", 0) or prompt_details.get("cache_read", 0)

        # Also try to extract from generations/message to get more details or as a fallback
        if response.generations and hasattr(response.generations[0][0], "message"):
            msg = response.generations[0][0].message
            
            # Check usage_metadata (standard LangChain structure)
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                input_tokens = input_tokens or um.get("input_tokens", 0)
                output_tokens = output_tokens or um.get("output_tokens", 0)
                cached_tokens = cached_tokens or um.get("cache_read_input_tokens") or um.get("cached_tokens") or 0
                
                input_details = um.get("input_token_details", {})
                if isinstance(input_details, dict):
                    cached_tokens = cached_tokens or input_details.get("cache_read", 0) or input_details.get("cached_tokens", 0)
                    
                output_details = um.get("output_token_details", {})
                if isinstance(output_details, dict):
                    reasoning_tokens = reasoning_tokens or output_details.get("reasoning", 0) or output_details.get("reasoning_tokens", 0)

            # Check response_metadata (raw API response structure)
            if hasattr(msg, "response_metadata") and msg.response_metadata:
                meta = msg.response_metadata
                usage_meta = meta.get("token_usage") or meta.get("usage")
                if isinstance(usage_meta, dict):
                    input_tokens = input_tokens or usage_meta.get("prompt_tokens", 0)
                    output_tokens = output_tokens or usage_meta.get("completion_tokens", 0)
                    cached_tokens = cached_tokens or usage_meta.get("cached_tokens") or usage_meta.get("cache_read_input_tokens") or usage_meta.get("cache_read") or 0
                    
                    p_details = usage_meta.get("prompt_tokens_details")
                    if isinstance(p_details, dict):
                        cached_tokens = cached_tokens or p_details.get("cached_tokens", 0) or p_details.get("cache_read", 0)
                        
                    o_details = usage_meta.get("completion_tokens_details")
                    if isinstance(o_details, dict):
                        reasoning_tokens = reasoning_tokens or o_details.get("reasoning_tokens", 0) or o_details.get("reasoning", 0)

        span_id = self.ctx.span_id.get()
        openai_processing_ms = self.ctx.openai_processing_ms_ledger.get(span_id, 0.0) if span_id else 0.0
        
        if not openai_processing_ms:
            if response.llm_output:
                openai_processing_ms = (
                    response.llm_output.get("openai-processing-ms") or 
                    response.llm_output.get("openai_processing_ms") or 0
                )
            
            if not openai_processing_ms and response.generations and hasattr(response.generations[0][0], "message"):
                meta = getattr(response.generations[0][0].message, "response_metadata", {})
                openai_processing_ms = (
                    meta.get("openai-processing-ms") or 
                    meta.get("openai_processing_ms") or 0
                )
                if not openai_processing_ms and "headers" in meta:
                    openai_processing_ms = meta["headers"].get("openai-processing-ms") or 0

        try:
            openai_processing_ms = float(openai_processing_ms)
        except (TypeError, ValueError):
            openai_processing_ms = 0

        network_latency_ms = max(0, latency - openai_processing_ms) if openai_processing_ms > 0 else 0

        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message"):
                        msg = gen.message
                        if hasattr(msg, "content") and msg.content:
                            if isinstance(msg.content, str):
                                output_bytes += len(msg.content.encode("utf-8"))
                            else:
                                output_bytes += len(json.dumps(msg.content).encode("utf-8"))
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            output_bytes += len(json.dumps(msg.tool_calls).encode("utf-8"))
                        elif hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("tool_calls"):
                            output_bytes += len(json.dumps(
                                msg.additional_kwargs["tool_calls"]
                            ).encode("utf-8"))
                    elif hasattr(gen, "text"):
                        if isinstance(gen.text, str):
                            output_bytes += len(gen.text.encode("utf-8"))
                        else:
                            output_bytes += len(json.dumps(gen.text).encode("utf-8"))

        self.metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "llm_call",
            **self.ctx.snapshot(),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id),
            "node_name": node_name,
            "latency_ms": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "wall_clock_s": round(latency / 1000, 4),
            "openai_processing_ms": round(openai_processing_ms, 2),
            "network_latency_ms": round(network_latency_ms, 2),
        }))
