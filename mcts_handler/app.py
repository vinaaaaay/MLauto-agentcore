from bedrock_agentcore.runtime import BedrockAgentCoreApp
from mcts_handler.handler import handle_action

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    return handle_action(payload)

if __name__ == "__main__":
    app.run()
