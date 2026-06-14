import logging
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from perception_agent.agent import build_perception_agent_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception_agent_app")

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    logger.info(f"Perception Agent invoked with payload keys: {list(payload.keys())}")
    graph = build_perception_agent_graph()
    initial_state = {
        "config": payload.get("config", {}),
        "input_data_folder": payload.get("input_data_folder", ""),
        "output_folder": payload.get("output_folder", "/tmp/perception_output"),
        "user_input": payload.get("user_input", ""),
        "all_error_analyses": payload.get("all_error_analyses", []),
    }
    result = graph.invoke(initial_state)
    return {
        "data_prompt": result.get("data_prompt", ""),
        "task_description": result.get("task_description", ""),
        "selected_tools": result.get("selected_tools", []),
        "current_tool": result.get("current_tool", ""),
        "tool_prompt": result.get("tool_prompt", ""),
    }

if __name__ == "__main__":
    app.run()
