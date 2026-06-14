from bedrock_agentcore.runtime import BedrockAgentCoreApp
from orchestrator.orchestrator import run_orchestration

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict):
    return run_orchestration(
        input_data_folder=payload.get("input_data_folder", ""),
        user_input=payload.get("user_input", ""),
        config=payload.get("config", {}),
        max_iterations=payload.get("max_iterations", 3),
        s3_bucket=payload.get("s3_bucket"),
        s3_uri=payload.get("s3_uri"),
        session_id=payload.get("session_id"),
        context_id=payload.get("context_id"),
        tracing=payload.get("tracing"),
    )

if __name__ == "__main__":
    app.run()
