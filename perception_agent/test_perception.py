import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add paths to sys.path
perception_agent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(perception_agent_dir))

# Load environment variables
local_env = perception_agent_dir / ".env"
if local_env.exists():
    print(f"Loading env from {local_env}")
    load_dotenv(local_env)
else:
    print("No local .env file found in perception_agent directory.")

from sandbox_client import SandboxClient
from app import handle

def run_extensive_perception_agent_test():
    print("=" * 70)
    print("  Extensive Perception Agent End-to-End Integration Test")
    print("=" * 70)

    # 1. Initialize SandboxClient
    print(">>> Step 1: Initializing Sandbox Client...")
    try:
        client = SandboxClient()
    except Exception as e:
        print(f"ERROR: Failed to initialize SandboxClient: {e}")
        sys.exit(1)

    # 2. Write mock data files to the sandbox
    print(">>> Step 2: Writing mock files to sandbox...")
    test_workspace = "/home/gem/workspace/test_perception_data"
    
    # Mock CSV data
    csv_data = """id,age,income,credit_score,approved
1,25,50000,650,0
2,45,120000,750,1
3,35,80000,710,1
4,22,25000,580,0
5,50,95000,680,1
"""
    
    # Mock README text
    readme_data = """# Credit Approval Classification Task
We want to train a model to predict whether a credit application is approved (the 'approved' column) based on the applicant's details: age, income, and credit score.
This is a standard binary classification problem.
Please use a tabular model like AutoGluon Tabular or scikit-learn.
"""
    
    csv_path = f"{test_workspace}/dataset.csv"
    readme_path = f"{test_workspace}/README.txt"
    dummy_path = f"{test_workspace}/unrelated_config.yaml"
    dummy_yaml = """mode: production\ndataset_name: credit_data\n"""

    print(f"    Writing {csv_path}...")
    s1 = client.write_file_sync(csv_path, csv_data)
    print(f"    Writing {readme_path}...")
    s2 = client.write_file_sync(readme_path, readme_data)
    print(f"    Writing {dummy_path}...")
    s3 = client.write_file_sync(dummy_path, dummy_yaml)

    if not (s1 and s2 and s3):
        print("ERROR: Failed to write test files inside the sandbox.")
        sys.exit(1)
    print("All test files successfully written to the sandbox.")

    # 3. Invoke Perception Agent entrypoint
    print(">>> Step 3: Invoking Perception Agent via entrypoint...")
    
    payload = {
        "input_data_folder": test_workspace,
        "output_folder": "/home/gem/workspace/test_perception_output",
        "user_input": "Find the description file, summarize the credit approval task, and select the best ML tool.",
        "config": {
            "sandbox_url": os.environ.get("SANDBOX_URL", "lambda:fame-sandbox-bastion")
        }
    }

    t0 = time.time()
    try:
        response = handle(payload)
        elapsed = time.time() - t0
        print(f"Perception agent pipeline executed in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"ERROR: Perception Agent invocation failed: {e}")
        # Clean up files before exiting
        client.exec_shell_sync(f"rm -rf {test_workspace}", cwd="")
        sys.exit(1)

    # 4. Validate Results
    print(">>> Step 4: Validating Response Structure and Outputs...")
    print("-" * 60)
    status = response.get("status")
    print(f"Status: {status}")
    print("-" * 60)

    # Check overall success
    if status != "COMPLETED":
        print(f"ERROR: Expected status to be 'COMPLETED', but got '{status}'.")
        print(f"Response error detail: {response.get('error')}")
        client.exec_shell_sync(f"rm -rf {test_workspace}", cwd="")
        sys.exit(1)

    # Validate output fields
    data_prompt = response.get("data_prompt", "")
    description_files = response.get("description_files", [])
    task_description = response.get("task_description", "")
    selected_tools = response.get("selected_tools", [])
    current_tool = response.get("current_tool", "")
    tool_prompt = response.get("tool_prompt", "")

    # Output validations
    print(f"Data Prompt length: {len(data_prompt)} chars")
    print(f"Description Files identified: {description_files}")
    print(f"Task Description length: {len(task_description)} chars")
    print(f"Selected Tools: {selected_tools}")
    print(f"Selected Current Tool: {current_tool}")
    print(f"Tool Prompt length: {len(tool_prompt)} chars")
    print("-" * 60)

    assert len(data_prompt) > 0, "Validation failed: data_prompt is empty."
    assert any("README.txt" in f for f in description_files), f"Validation failed: README.txt not identified in {description_files}."
    assert len(task_description) > 0, "Validation failed: task_description is empty."
    assert len(selected_tools) > 0, "Validation failed: selected_tools list is empty."
    assert len(current_tool) > 0, "Validation failed: current_tool is not selected."
    assert len(tool_prompt) > 0, "Validation failed: tool_prompt is empty."

    print("All validations PASSED successfully!")

    # 5. Clean up sandbox
    print(">>> Step 5: Cleaning up test files in sandbox...")
    client.exec_shell_sync(f"rm -rf {test_workspace}", cwd="")
    print("Test cleanup complete.")
    print("=" * 70)
    print("Perception Agent end-to-end integration test successfully verified!")
    print("=" * 70)

if __name__ == "__main__":
    run_extensive_perception_agent_test()
