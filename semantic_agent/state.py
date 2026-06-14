import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict, Any, TypedDict

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════

class TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class SemanticAgentState(TypedDict):
    """
    State representing the data flow through the Semantic Agent graph.
    Supports a flexible dictionary model to easily support additional parameters.
    """
    # Configuration
    config: Dict[str, Any]
    output_folder: str

    # Context & Inputs
    task_description: str
    data_prompt: str
    user_input: str
    all_error_analyses: List[str]
    current_tool: str

    # Outputs
    search_query: str
    tutorial_retrieval: List[TutorialInfo]
    tutorial_prompt: str


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Client
# ═══════════════════════════════════════════════════════════════════════════

class VectorStoreMCPClient:
    """
    Client for interacting with the Vector Store MCP Server.
    """
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    async def retrieve_tutorials(
        self,
        query: str,
        tool_name: str,
        top_k: int = 5,
        condensed: bool = False,
        callbacks: list = None
    ) -> List[Dict[str, Any]]:
        """
        Call the retrieve_tutorials tool on the MCP server with callback hooks.
        """
        import uuid
        run_id = uuid.uuid4()
        parent_run_id = uuid.uuid4()

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_tool_start"):
                    try:
                        cb.on_tool_start(
                            {"name": "mcp_retrieve_tutorials"},
                            json.dumps({
                                "query": query,
                                "tool_name": tool_name,
                                "top_k": top_k,
                                "condensed": condensed
                            }),
                            run_id=run_id,
                            parent_run_id=parent_run_id,
                        )
                    except Exception as cb_err:
                        logger.warning(f"Failed to call on_tool_start callback: {cb_err}")

        try:
            raw_tutorials = await self._retrieve_tutorials_impl(
                query=query,
                tool_name=tool_name,
                top_k=top_k,
                condensed=condensed
            )
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_tool_end"):
                        try:
                            cb.on_tool_end(
                                json.dumps({"tutorial_count": len(raw_tutorials)}),
                                run_id=run_id,
                                parent_run_id=parent_run_id,
                            )
                        except Exception as cb_err:
                            logger.warning(f"Failed to call on_tool_end callback: {cb_err}")
            return raw_tutorials
        except Exception as e:
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_tool_error"):
                        try:
                            cb.on_tool_error(
                                e,
                                run_id=run_id,
                                parent_run_id=parent_run_id,
                            )
                        except Exception as cb_err:
                            logger.warning(f"Failed to call on_tool_error callback: {cb_err}")
            raise

    async def _retrieve_tutorials_impl(
        self,
        query: str,
        tool_name: str,
        top_k: int = 5,
        condensed: bool = False
    ) -> List[Dict[str, Any]]:
        import httpx
        # Build direct POST URLs to try
        urls_to_try = []
        if self.server_url.endswith("/invocations") or self.server_url.endswith("/retrieve_tutorials"):
            urls_to_try.append(self.server_url)
        else:
            urls_to_try.append(f"{self.server_url}/retrieve_tutorials")
            urls_to_try.append(f"{self.server_url}/invocations")

        import httpx
        payload_data = {
            "query": query,
            "tool_name": tool_name,
            "top_k": top_k,
            "condensed": condensed
        }

        for post_url in urls_to_try:
            try:
                logger.info(f"Attempting direct HTTP POST to {post_url} (timeout=120.0s)...")
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(post_url, json=payload_data)
                    logger.info(f"Direct HTTP POST to {post_url} responded with status_code={resp.status_code}")
                    if resp.status_code == 200:
                        data = resp.json()
                        # Automatically unpack Mangum integration response if returned as a raw dictionary
                        if isinstance(data, dict) and "statusCode" in data and "body" in data:
                            logger.info("Unpacking Mangum response format...")
                            return json.loads(data["body"])
                        return data
                    else:
                        logger.warning(f"Direct HTTP POST to {post_url} non-200 response: {resp.status_code} - {resp.text[:500]}")
            except Exception as e:
                logger.warning(f"Direct HTTP POST to {post_url} failed: {e!r}")

        logger.warning("All direct HTTP POST endpoints failed. Falling back to standard MCP SSE...")

        from mcp import ClientSession
        from mcp.client.sse import sse_client

        sse_url = f"{self.server_url}/sse"
        async with sse_client(sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    name="retrieve_tutorials",
                    arguments={
                        "query": query,
                        "tool_name": tool_name,
                        "top_k": top_k,
                        "condensed": condensed,
                    }
                )
                
                raw_tutorials = []
                if result.content:
                    for block in result.content:
                        try:
                            text_val = block.text.strip()
                            if text_val.startswith("{") or text_val.startswith("["):
                                parsed = json.loads(text_val)
                                if isinstance(parsed, list):
                                    raw_tutorials.extend(parsed)
                                else:
                                    raw_tutorials.append(parsed)
                            else:
                                logger.warning(f"Non-JSON block content ignored: {text_val[:100]}")
                        except Exception as e:
                            logger.warning(f"Failed to parse block as JSON: {e}")
                return raw_tutorials
