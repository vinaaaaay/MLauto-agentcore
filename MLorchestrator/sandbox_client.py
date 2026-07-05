import base64
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
import boto3
import asyncio

logger = logging.getLogger(__name__)

def run_sync(coro):
    """Run an async coroutine synchronously."""
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


class SandboxClient:
    """
    Synchronous Sandbox Client communicating with the MCP Server on private EC2 instances
    proxied through the AWS Lambda bastion gateway.
    """

    def __init__(self, sandbox_url: Optional[str] = None):
        url = sandbox_url or os.environ.get("SANDBOX_URL")
        self.gateway_lambda_name = os.environ.get("GATEWAY_LAMBDA_NAME")
        if url and (url.startswith("lambda:") or url.startswith("arn:aws:lambda:")):
            parts = url.split(":")
            self.gateway_lambda_name = parts[-1].split("?")[0]
            
        if not self.gateway_lambda_name:
            self.gateway_lambda_name = "fame-sandbox-bastion"

        self.target_ip = os.environ.get("TARGET_IP", "172.31.41.84")
        self.target_port = int(os.environ.get("TARGET_PORT", "8080"))
        self.region_name = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        
        logger.info(
            f"SandboxClient operating in Lambda Gateway mode. "
            f"Target: {self.gateway_lambda_name} -> {self.target_ip}:{self.target_port}"
        )
        from botocore.config import Config
        config = Config(
            read_timeout=900,
            connect_timeout=60,
            retries={'max_attempts': 1}
        )
        self.lambda_client = boto3.client("lambda", region_name=self.region_name, config=config)
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _invoke_lambda_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously invokes the gateway Lambda."""
        response = self.lambda_client.invoke(
            FunctionName=self.gateway_lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        payload_data = response["Payload"].read().decode("utf-8")
        return json.loads(payload_data)

    async def _invoke_gateway_async(self, method: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the gateway Lambda asynchronously in a thread."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        auth_key = os.environ.get("SANDBOX_MCP_AUTH_KEY")
        if auth_key:
            auth_key_clean = auth_key.strip('"')
            headers["Authorization"] = f"Bearer {auth_key_clean}"

        gateway_payload = {
            "target_ip": self.target_ip,
            "target_port": self.target_port,
            "method": method,
            "path": path,
            "headers": headers,
            "body": body
        }
        
        lambda_resp = await asyncio.to_thread(self._invoke_lambda_sync, gateway_payload)
        status_code = lambda_resp.get("statusCode", 500)
        resp_body = lambda_resp.get("body")
        
        if status_code != 200:
            err_msg = f"Gateway Lambda returned status {status_code}: {resp_body}"
            logger.error(err_msg)
            return {"error": {"code": status_code, "message": err_msg}}
            
        if isinstance(resp_body, dict):
            return resp_body
        elif isinstance(resp_body, str):
            try:
                return json.loads(resp_body)
            except json.JSONDecodeError:
                return self._parse_sse_response(resp_body)
        else:
            return {"error": {"code": -32000, "message": f"Unexpected body type: {type(resp_body)}"}}

    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"error": {"code": -32000, "message": "Empty response from bastion"}}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        last_result = {}
        current_event_type = "message"
        data_lines = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                current_event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            elif line == "":
                raw = "\n".join(data_lines).strip()
                if raw and current_event_type != "log":
                    try:
                        obj = json.loads(raw)
                        if "result" in obj or "error" in obj:
                            last_result = obj
                    except json.JSONDecodeError:
                        pass
                data_lines = []
        
        raw = "\n".join(data_lines).strip()
        if raw and current_event_type != "log":
            try:
                obj = json.loads(raw)
                if "result" in obj or "error" in obj:
                    last_result = obj
            except json.JSONDecodeError:
                pass
                
        return last_result if last_result else {"error": {"code": -32000, "message": f"Failed to parse SSE payload: {text[:200]}"}}

    async def exec_shell(self, command: str, cwd: str = "/home/gem/workspace", timeout: Optional[int] = None) -> Tuple[bool, str, str]:
        """Executes a command inside the sandbox synchronously."""
        full_command = f"mkdir -p {cwd} && cd {cwd} && {command}" if cwd else command
        encoded_cmd = base64.b64encode(full_command.encode("utf-8")).decode("utf-8")
        wrapper_command = f"echo -n '{encoded_cmd}' | base64 -d | bash"

        args = {"cmd": wrapper_command, "cwd": "/tmp"}
        if timeout is not None:
            args["timeout"] = timeout
        res = await self.call_tool("sandbox_execute_bash", args)
        inner = self._parse_inner_response(res)
        
        if "error" in inner:
            return False, "", str(inner["error"])
            
        exit_code = inner.get("exit_code", 0)
        output = inner.get("output", "")
        if exit_code == 0:
            return True, output, ""
        return False, "", output or f"Command exited with code {exit_code}"

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes a tool on the sandbox MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_id()
        }
        return await self._invoke_gateway_async("POST", "/mcp", payload)

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

    # ── Synchronous wrappers ──

    def exec_shell_sync(self, command: str, cwd: str = "/home/gem/workspace", timeout: Optional[int] = None) -> Tuple[bool, str, str]:
        return run_sync(self.exec_shell(command, cwd, timeout))
