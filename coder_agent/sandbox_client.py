"""
Sandbox Client for Coder Agent.

Communicates with the EC2-hosted sandbox MCP server via SSE transport
using the `mcp` Python SDK.

The MCP server (sandbox_mcp/main.py) exposes:
  HTTP  POST /reset_sandbox  — Resets the sandbox container
  HTTP  GET  /sse            — Establishes the SSE MCP connection
  HTTP  POST /messages/      — Receives MCP client messages

MCP Tools (called over the SSE connection):
  exec_sandbox   — Executes a bash command; task-based (delivery=poll)
  write_file     — Writes a file to an absolute path in the container
  read_file      — Reads a file from an absolute path in the container
  kill_processes — Kills sandbox processes for a node_id or iteration

Environment variables:
    SANDBOX_URL          — Base URL of the MCP server (e.g. http://13.x.x.x:8081)
    SANDBOX_MCP_URL      — Full SSE URL override (e.g. http://13.x.x.x:8081/sse)
    SANDBOX_MCP_AUTH_KEY — Bearer token for the MCP server (if auth is enabled)
    SANDBOX_TIMEOUT      — Default command timeout in seconds (default: 1800)
    MCP_TASK_TTL_MS      — MCP task TTL in milliseconds (default: 3600000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MCP_TASK_TTL_MS = int(os.environ.get("MCP_TASK_TTL_MS", str(3_600_000)))  # 1 hour
_POLL_INTERVAL = 2  # seconds between task-status polls
_DEFAULT_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "1800"))  # 30 min

_TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}


def _build_sse_url(sandbox_url: Optional[str] = None) -> str:
    """Derive the /sse endpoint from whatever URL variant is configured."""
    url = (
        sandbox_url
        or os.environ.get("SANDBOX_MCP_URL", "")
        or os.environ.get("SANDBOX_URL", "")
    )
    if not url:
        raise ValueError(
            "No sandbox URL configured. Set SANDBOX_URL or SANDBOX_MCP_URL."
        )
    url = url.rstrip("/")
    if url.endswith("/sse"):
        return url
    if url.endswith("/mcp"):          # legacy path — replace suffix
        return url[: -len("/mcp")] + "/sse"
    return url + "/sse"


def run_sync(coro):
    """Run an async coroutine synchronously (safe inside running loops via nest_asyncio)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    return loop.run_until_complete(coro)


# ── SandboxClient ─────────────────────────────────────────────────────────────

class SandboxClient:
    """
    Async sandbox client that speaks the MCP protocol over SSE.

    A single persistent SSE connection is lazily established on the first call
    and reused for the lifetime of the client — important for exec_sandbox tasks
    which may run for up to 30 minutes while the client polls task status.

    Context-manager usage (recommended):
        async with SandboxClient(url) as sandbox:
            ok = await sandbox.write_file("/tmp/foo.py", code)
            success, out, err = await sandbox.exec_shell("python /tmp/foo.py")

    Or lazy-connect (LangGraph state):
        sandbox = SandboxClient(url)           # no connection yet
        await sandbox.write_file(...)          # connects on first call
        ...
        await sandbox.close()                  # call in finally block
    """

    def __init__(self, sandbox_url: Optional[str] = None):
        self._sse_url = _build_sse_url(sandbox_url)
        self._session = None
        self._sse_ctx = None
        self._session_ctx = None
        self._connected = False
        logger.info(f"SandboxClient created — SSE endpoint: {self._sse_url}")

    # ── Connection lifecycle ───────────────────────────────────────────────

    async def _ensure_connected(self):
        if self._connected:
            return

        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        headers = {}
        auth_key = os.environ.get("SANDBOX_MCP_AUTH_KEY", "").strip('"')
        if auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"

        self._sse_ctx = sse_client(self._sse_url, headers=headers)
        self._read, self._write = await self._sse_ctx.__aenter__()
        self._session_ctx = ClientSession(self._read, self._write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        self._connected = True
        logger.info(f"SandboxClient SSE session established to {self._sse_url}")

    async def close(self):
        """Cleanly tear down the SSE session."""
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
        if self._sse_ctx:
            try:
                await self._sse_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._sse_ctx = None
        self._session = None
        self._connected = False
        logger.info("SandboxClient SSE session closed.")

    async def __aenter__(self) -> "SandboxClient":
        await self._ensure_connected()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _parse_tool_result(self, result) -> dict:
        """Extract the JSON payload from an MCP CallToolResult."""
        try:
            for block in getattr(result, "content", []):
                if getattr(block, "type", None) == "text":
                    return json.loads(block.text)
        except (json.JSONDecodeError, Exception):
            pass
        return {"raw": str(result)}

    async def _poll_task_until_done(
        self, task_id: str, poll_timeout: int = _DEFAULT_TIMEOUT
    ) -> dict:
        """
        Poll an MCP task (delivery=poll exec_sandbox) until it reaches a
        terminal state, then fetch and return the result payload.
        """
        from mcp.types import CallToolResult

        start = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > poll_timeout:
                raise TimeoutError(
                    f"exec_sandbox task {task_id} did not complete "
                    f"within {poll_timeout}s"
                )
            try:
                task_resp = await self._session.experimental.get_task(task_id)
            except Exception:
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            if task_resp.status in _TERMINAL_TASK_STATES:
                break
            await asyncio.sleep(_POLL_INTERVAL)

        result = await self._session.experimental.get_task_result(
            task_id, CallToolResult
        )
        logger.info(f"[poll_task] task_id={task_id} completed")
        return self._parse_tool_result(result)

    # ── Public MCP tool wrappers ───────────────────────────────────────────

    async def write_file(self, path: str, content: str) -> bool:
        """Write content to an absolute path inside the sandbox container."""
        await self._ensure_connected()
        result = await self._session.call_tool(
            "write_file", {"path": path, "content": content}
        )
        parsed = self._parse_tool_result(result)
        success = parsed.get("success", False)
        if not success:
            logger.error(f"write_file failed for {path}: {parsed}")
        return success

    async def read_file(self, path: str) -> str:
        """Read the contents of an absolute path inside the sandbox container."""
        await self._ensure_connected()
        result = await self._session.call_tool("read_file", {"path": path})
        parsed = self._parse_tool_result(result)
        if not parsed.get("success", True):
            raise IOError(f"Failed to read file {path}: {parsed}")
        return parsed.get("content", "")

    async def exec_command(
        self, command: str, timeout: Optional[int] = None
    ) -> dict:
        """
        Execute a bash command in the sandbox via the exec_sandbox MCP task.

        Uses delivery=poll: the MCP server starts the command and the client
        polls task status every 2 seconds over the persistent SSE connection.
        Blocks until the command finishes or `timeout` is exceeded.

        Returns a dict with keys: stdout, stderr, exit_code.
        """
        await self._ensure_connected()
        effective_timeout = timeout or _DEFAULT_TIMEOUT
        args: dict = {
            "command": command,
            "delivery": "poll",
            "timeout": effective_timeout,
        }

        try:
            create = await self._session.experimental.call_tool_as_task(
                "exec_sandbox", args, ttl=MCP_TASK_TTL_MS
            )
            task_id = create.task.taskId
            logger.info(
                f"[exec_command] task created: task_id={task_id} "
                f"cmd={command[:80]!r} timeout={effective_timeout}s"
            )
        except Exception as e:
            logger.error(
                f"[exec_command] call_tool_as_task failed ({e}); "
                "falling back to synchronous call_tool"
            )
            fallback = await self._session.call_tool("exec_sandbox", args)
            return self._parse_tool_result(fallback)

        return await self._poll_task_until_done(task_id, poll_timeout=effective_timeout)

    async def exec_shell(
        self,
        command: str,
        cwd: str = "/home/gem/workspace",
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute a shell command inside the sandbox, optionally cd-ing to `cwd`.

        Blocks until the command finishes (Option A — blocking MCP task poll).
        Returns (success: bool, stdout: str, stderr: str).
        """
        full_command = (
            f"mkdir -p {cwd} && cd {cwd} && {command}" if cwd else command
        )
        result = await self.exec_command(full_command, timeout=timeout)
        exit_code = result.get("exit_code", -1)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        success = exit_code == 0
        if not success:
            logger.warning(
                f"exec_shell exited with code {exit_code}: cmd={command[:60]!r}"
            )
        return success, stdout, stderr or (
            f"Command exited with code {exit_code}" if not success else ""
        )

    async def kill_iteration_processes(
        self,
        node_id: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> dict:
        """Kill sandbox processes associated with a node_id or iteration."""
        await self._ensure_connected()
        args: dict = {}
        if node_id is not None:
            args["node_id"] = node_id
        if iteration is not None:
            args["iteration"] = iteration
        result = await self._session.call_tool("kill_processes", args)
        return self._parse_tool_result(result)

    # ── Synchronous wrappers (for callers outside async context) ──────────

    def read_file_sync(self, path: str) -> str:
        return run_sync(self.read_file(path))

    def write_file_sync(self, path: str, content: str) -> bool:
        return run_sync(self.write_file(path, content))

    def exec_shell_sync(
        self,
        command: str,
        cwd: str = "/home/gem/workspace",
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str, str]:
        return run_sync(self.exec_shell(command, cwd, timeout))
