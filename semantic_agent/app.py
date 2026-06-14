import asyncio
import logging
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from semantic_agent.agent import build_semantic_agent_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_agent_app")

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    logger.info(f"Semantic Agent invoked with payload keys: {list(payload.keys())}")
    graph = build_semantic_agent_graph()
    initial_state = {
        "config": payload.get("config", {}),
        "task_description": payload.get("task_description", ""),
        "data_prompt": payload.get("data_prompt", ""),
        "user_input": payload.get("user_input", ""),
        "current_tool": payload.get("current_tool", ""),
        "all_error_analyses": payload.get("all_error_analyses", []),
        "output_folder": payload.get("output_folder", "/tmp/semantic_output"),
    }
    
    result = asyncio.run(graph.ainvoke(initial_state))
    
    return {"tutorial_prompt": result.get("tutorial_prompt", "")}

if __name__ == "__main__":
    app.run()
