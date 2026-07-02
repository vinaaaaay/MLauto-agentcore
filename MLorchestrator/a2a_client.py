import logging
import os
import time
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class A2AClientError(Exception):
    """Base exception for A2A client errors."""
    pass

class A2AClient:
    """
    Client for communicating natively with Bedrock AgentCore agents using standard JSON-in/JSON-out.
    Directly calls boto3's invoke_agent_runtime synchronously.
    """
    
    def __init__(self, agent_url: str, agent_name: str = "agent"):
        self.agent_url = agent_url.strip()
        self.agent_name = agent_name
        self.call_logger = None  # Will be set by Orchestrator if active
    
    def send_task_sync(self, skill: str, data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Synchronously send a task to the AgentCore agent using standard JSON invocation.

        Args:
            skill: Skill name sent in the payload (also used for logging).
            data: Payload dict merged with the skill key.
            session_id: Optional AgentCore runtime session ID. When provided,
                the request is routed to the existing warm session instead of
                spawning a new one. Use this to pin polling calls to the same
                container (e.g. coder agent long-running tasks).
        """
        import boto3
        from botocore.config import Config
        
        start_time = time.time()
        logger.info(f"Invoking Bedrock AgentCore agent: {self.agent_url} (skill={skill})")
        region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
        
        # Configure client with increased read timeout (e.g., 30 minutes) and no retries
        # to prevent client-side timeout loops when agents take longer than 60 seconds.
        config = Config(
            read_timeout=1800,
            connect_timeout=60,
            retries={'max_attempts': 1}
        )
        client = boto3.client("bedrock-agentcore", region_name=region, config=config)
        
        # Build the payload directly
        payload = {"skill": skill, **data}
        
        invoke_params = {
            "agentRuntimeArn": self.agent_url,
            "qualifier": "DEFAULT",
            "contentType": "application/json",
            "accept": "application/json",
            "payload": json.dumps(payload).encode(),
        }
        if session_id:
            invoke_params["runtimeSessionId"] = session_id
            logger.info(f"Using session_id={session_id} for sticky routing.")
        
        try:
            response = client.invoke_agent_runtime(**invoke_params)
            elapsed = time.time() - start_time
            
            body = response.get("response") or response.get("body")
            if hasattr(body, "read"):
                body_bytes = body.read()
            else:
                body_bytes = body
                
            text = body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else str(body_bytes)
            result_data = json.loads(text)
            
            if self.call_logger:
                self.call_logger.log_call(
                    self.agent_name, skill, payload, result_data, elapsed
                )
                
            return result_data

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error communicating with AgentCore agent at {self.agent_url}: {e}")
            raise A2AClientError(f"Failed to communicate with AgentCore agent: {e}") from e
