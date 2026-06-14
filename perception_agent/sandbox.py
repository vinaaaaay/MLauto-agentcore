import abc
import json
import boto3
import os
import logging
import asyncio
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_sync(coro):
    """
    Run an async coroutine synchronously.
    Handles situations where an event loop is already running (e.g. inside Starlette/LangGraph context).
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)


class BaseSandboxClient(abc.ABC):
    """Abstract interface for all agent execution environments."""

    @abc.abstractmethod
    async def read_file(self, path: str) -> str:
        """Reads a file from the sandbox."""
        pass

    @abc.abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """Writes content to a file in the sandbox."""
        pass

    @abc.abstractmethod
    async def exec_shell(self, command: str, cwd: str = "/home/gem/workspace") -> Tuple[bool, str, str]:
        """
        Executes a command, streams output to stdout, and returns the final state.
        Returns: (success_boolean, complete_stdout, complete_stderr)
        """
        pass


class BastionSandboxClient(BaseSandboxClient):
    """
    Adapter that communicates with the MCP Server on private EC2 instances 
    proxied through the AWS Lambda bastion gateway.
    
    Implements BaseSandboxClient asynchronously by running blocking boto3 Lambda calls 
    in a thread pool via asyncio.to_thread.
    """

    def __init__(
        self,
        gateway_lambda_name: Optional[str] = None,
        target_ip: Optional[str] = None,
        target_port: Optional[int] = None,
        region_name: Optional[str] = None
    ):
        self.gateway_lambda_name = gateway_lambda_name or os.environ.get("GATEWAY_LAMBDA_NAME", "fame-sandbox-bastion")
        self.target_ip = target_ip or os.environ.get("TARGET_IP", "172.31.41.59")
        self.target_port = target_port or int(os.environ.get("TARGET_PORT", "8080"))
        
        # Initialize boto3 client.
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        self.lambda_client = boto3.client("lambda", region_name=self.region_name)
        self._request_id = 0
        logger.info(
            f"Initialized BastionSandboxClient targeting {self.target_ip}:{self.target_port} "
            f"via Lambda {self.gateway_lambda_name} in region {self.region_name}"
        )

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _invoke_lambda_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously invokes the gateway Lambda. Run this via asyncio.to_thread."""
        logger.debug(f"Invoking Gateway Lambda: {self.gateway_lambda_name} with payload keys {list(payload.keys())}")
        response = self.lambda_client.invoke(
            FunctionName=self.gateway_lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        payload_data = response["Payload"].read().decode("utf-8")
        return json.loads(payload_data)

    async def _invoke_gateway(self, method: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Wraps Lambda invocation in a thread and invokes it asynchronously."""
        gateway_payload = {
            "target_ip": self.target_ip,
            "target_port": self.target_port,
            "method": method,
            "path": path,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            "body": body
        }
        
        # Invoke via thread pool to keep loop non-blocking
        lambda_resp = await asyncio.to_thread(self._invoke_lambda_sync, gateway_payload)
        
        # Check lambda execution status
        status_code = lambda_resp.get("statusCode", 500)
        resp_body = lambda_resp.get("body")
        
        if status_code != 200:
            err_msg = f"Gateway Lambda returned status {status_code}: {resp_body}"
            logger.error(err_msg)
            return {"error": {"code": status_code, "message": err_msg}}
            
        # The body might be a parsed dict or a string (possibly SSE)
        if isinstance(resp_body, dict):
            return resp_body
        elif isinstance(resp_body, str):
            # Parse as SSE or plain JSON
            return self._parse_mcp_response(resp_body)
        else:
            return {"error": {"code": -32000, "message": f"Unexpected body type: {type(resp_body)}"}}

    def _parse_mcp_response(self, text: str) -> Dict[str, Any]:
        """Parses the MCP response which might be SSE formatted text or plain JSON."""
        text = text.strip()
        if not text:
            return {"error": {"code": -32000, "message": "Empty response from bastion"}}
            
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Otherwise, parse as SSE stream
        last_result: Dict[str, Any] = {}
        current_event_type = "message"
        data_lines = []
        
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                current_event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            elif line == "":
                # Flush event
                raw = "\n".join(data_lines).strip()
                if raw:
                    if current_event_type == "log":
                        logger.info(f"[sandbox log] {raw}")
                    else:
                        try:
                            obj = json.loads(raw)
                            if "result" in obj or "error" in obj:
                                last_result = obj
                        except json.JSONDecodeError:
                            pass
                current_event_type = "message"
                data_lines = []
                
        # Flush final event if any
        raw = "\n".join(data_lines).strip()
        if raw and current_event_type != "log":
            try:
                obj = json.loads(raw)
                if "result" in obj or "error" in obj:
                    last_result = obj
            except json.JSONDecodeError:
                pass
                
        if last_result:
            return last_result
            
        return {"error": {"code": -32000, "message": f"Failed to parse SSE payload: {text[:200]}"}}

    def _parse_inner_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in response:
            return response
        content_list = response.get("result", {}).get("content", [])
        if not content_list:
            return response
        text = content_list[0].get("text", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_output": text}

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper around calling a tool on the EC2 MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_id()
        }
        return await self._invoke_gateway("POST", "/mcp", payload)

    async def read_file(self, path: str) -> str:
        """Reads a file from the sandbox."""
        res = await self.call_tool("sandbox_file_operations", {"action": "read", "path": path})
        inner = self._parse_inner_response(res)
        if "error" in inner:
            raise IOError(f"Failed to read file {path}: {inner['error']}")
        return inner.get("content", "")

    async def write_file(self, path: str, content: str) -> bool:
        """Writes content to a file in the sandbox, with auto-mkdir and chunking."""
        parent_dir = os.path.dirname(path)
        if parent_dir:
            await self.exec_shell(f"mkdir -p {parent_dir}", cwd="")
        await self.exec_shell(f"rm -f {path}", cwd="")
        
        chunk_size = 50000
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            res = await self.call_tool("sandbox_file_operations", {
                "action": "write",
                "path": path,
                "content": chunk,
                "append": True,
            })
            inner = self._parse_inner_response(res)
            if "error" in inner:
                logger.error(f"Error writing chunk to {path}: {inner['error']}")
                return False
            # Check if write was successful
            if not inner.get("success", False) and inner.get("action") != "write":
                logger.error(f"Unexpected non-success response: {inner}")
                return False
        return True

    async def exec_shell(self, command: str, cwd: str = "/home/gem/workspace") -> Tuple[bool, str, str]:
        """Executes a command inside the sandbox."""
        import base64
        # Automatically ensure cwd exists and navigate to it inside the shell,
        # passing /tmp as the tool's execution directory to prevent tool-level startup failures.
        full_command = f"mkdir -p {cwd} && cd {cwd} && {command}" if cwd else command
        encoded_cmd = base64.b64encode(full_command.encode("utf-8")).decode("utf-8")
        wrapper_command = f"echo -n '{encoded_cmd}' | base64 -d | bash"

        res = await self.call_tool("sandbox_execute_bash", {"cmd": wrapper_command, "cwd": "/tmp"})
        inner = self._parse_inner_response(res)
        
        if "error" in inner:
            return False, "", str(inner["error"])
            
        exit_code = inner.get("exit_code", 0)
        output = inner.get("output", "")
        if exit_code == 0:
            return True, output, ""
        return False, "", output or f"Command exited with code {exit_code}"

    # ── Synchronous wrappers ──

    def read_file_sync(self, path: str) -> str:
        """Reads a file synchronously from the sandbox."""
        return run_sync(self.read_file(path))

    def write_file_sync(self, path: str, content: str) -> bool:
        """Writes content to a file synchronously in the sandbox."""
        return run_sync(self.write_file(path, content))

    def exec_shell_sync(self, command: str, cwd: str = "/home/gem/workspace") -> Tuple[bool, str, str]:
        """Executes a command synchronously inside the sandbox."""
        return run_sync(self.exec_shell(command, cwd))
