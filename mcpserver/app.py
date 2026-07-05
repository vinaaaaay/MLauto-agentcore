"""
MCP Vector Store Server — Bedrock AgentCore Entrypoint.

Wraps the retrieve_tutorials function from mcp_server.py in a
BedrockAgentCoreApp harness, matching the deployment pattern used by the
react_distributed agents (planner, actor, evaluator, orchestrator).

Payload schema (JSON):
    {
        "query":      str,                # required — semantic search query
        "tool_name":  str,                # required — tool/library name to search
        "top_k":      int  (default 5),   # optional — number of results to return
        "condensed":  bool (default False) # optional — use condensed tutorials
    }

Response schema (JSON):
    {
        "results": [
            {
                "path":    str,
                "title":   str,
                "summary": str,
                "score":   float,
                "content": str
            },
            ...
        ],
        "result_count": int,
        "query":        str,
        "tool_name":    str
    }
"""

import json
import logging
import os
import sys
from pathlib import Path

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ─── ensure mcp_server module is importable from the same directory ───────────
_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from mcp_server import retrieve_tutorials  # noqa: E402

# ─── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_agentcore")

# Structured metrics logger
metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    metric_logger.addHandler(_h)

from common_local.metrics_context import MetricsContext
from common_local.metrics_emitter import emit_event, graph_metrics

ctx = MetricsContext(agent_id="mcpserver")

# ─── AgentCore App ────────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()


@app.entrypoint
@graph_metrics(ctx=ctx, logger=metric_logger, graph_name="mcpserver")
def handle(payload: dict) -> dict:
    """
    AgentCore entrypoint — receives a JSON payload and returns retrieval results.

    Called by AgentCore's runtime harness on every invocation.
    """
    query = payload.get("query", "")
    tool_name = payload.get("tool_name", "")
    top_k = int(payload.get("top_k", 5))
    condensed = bool(payload.get("condensed", False))

    if not query:
        logger.warning("Received payload with empty 'query'. Returning empty results.")
        return {"results": [], "result_count": 0, "query": query, "tool_name": tool_name}

    if not tool_name:
        logger.warning("Received payload with empty 'tool_name'. Returning empty results.")
        return {"results": [], "result_count": 0, "query": query, "tool_name": tool_name}

    logger.info(f"[handle] query={query[:80]!r}, tool_name={tool_name!r}, top_k={top_k}, condensed={condensed}")

    try:
        results = retrieve_tutorials(
            query=query,
            tool_name=tool_name,
            top_k=top_k,
            condensed=condensed,
        )
        import resource
        import time
        peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        logger.info(f"[handle] returned {len(results)} result(s) | Peak RAM: {peak_ram_mb:.2f} MB")
        return {
            "results": results,
            "result_count": len(results),
            "query": query,
            "tool_name": tool_name,
        }

    except Exception as exc:
        error_info = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "query": query,
            "tool_name": tool_name,
        }
        logger.error(f"[handle] exception: {json.dumps(error_info)}")
        emit_event(metric_logger, {
            **ctx.snapshot(),
            "event_type": "invocation",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        })
        raise


if __name__ == "__main__":
    from dotenv import load_dotenv
    # Load .env from the same directory as this file (mcpserver/.env)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    app.run()
