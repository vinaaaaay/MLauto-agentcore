import os
import sys
from pathlib import Path

# Force Hugging Face to use the bundled hf_home directory
os.environ["HF_HOME"] = str(Path(__file__).parent / "hf_home")

import logging
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcp_server import retrieve_tutorials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_store_agentcore")

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    logger.info(f"Vector Store AgentCore invoked with payload keys: {list(payload.keys())}")
    query = payload.get("query")
    tool_name = payload.get("tool_name")
    top_k = payload.get("top_k", 5)
    condensed = payload.get("condensed", False)
    
    return retrieve_tutorials(
        query=query,
        tool_name=tool_name,
        top_k=top_k,
        condensed=condensed
    )

if __name__ == "__main__":
    app.run()
