import asyncio
import logging
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from coder_agent.agent import build_coder_agent_graph
from coder_agent.tools import BastionSandboxClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coder_agent_app")

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    logger.info(f"Coder Agent invoked with payload keys: {list(payload.keys())}")
    config = payload.get("config", {})
    graph = build_coder_agent_graph()
    initial_state = {
        "config": config,
        "task_description": payload.get("task_description", ""),
        "data_prompt": payload.get("data_prompt", ""),
        "user_input": payload.get("user_input", ""),
        "current_tool": payload.get("current_tool", ""),
        "tool_prompt": payload.get("tool_prompt", ""),
        "tutorial_prompt": payload.get("tutorial_prompt", ""),
        "all_error_analyses": payload.get("all_error_analyses", []),
        "previous_python_code": payload.get("previous_python_code", ""),
        "previous_bash_script": payload.get("previous_bash_script", ""),
        "stage": payload.get("stage", "root"),
        "iteration": payload.get("iteration", 0),
        "node_id": payload.get("node_id"),
        "output_folder": payload.get("output_folder", "/tmp/coder_output"),
        "sandbox_client": BastionSandboxClient(),
    }
    
    result = asyncio.run(graph.ainvoke(initial_state))
    
    return {
        "python_code": result.get("python_code", ""),
        "bash_script": result.get("bash_script", ""),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "decision": result.get("decision", "FIX"),
        "error_summary": result.get("error_summary", ""),
        "validation_score": result.get("validation_score"),
        "error_message": result.get("error_message", ""),
    }

if __name__ == "__main__":
    app.run()
