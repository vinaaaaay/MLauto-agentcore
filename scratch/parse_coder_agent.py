import json
from pathlib import Path

run_dir = Path("/home/administrator/dreamlab/MLauto-agentcore/runs/run_21")
coder_json_path = run_dir / "coder_agent.json"

with open(coder_json_path, "r") as f:
    data = json.load(f)

print(f"Total elements: {len(data)}")

# Let's group by skill or trace how invocations work.
# An invocation can start with a "generate_and_run" and then have multiple "check_status" calls until it's finished?
# Let's print each element's skill, action, and key fields in input/output.
for idx, item in enumerate(data):
    skill = item.get("skill")
    action = item.get("input", {}).get("action")
    node_id = item.get("input", {}).get("node_id") or item.get("input", {}).get("job_id")
    status = item.get("output", {}).get("status")
    
    # We want to see how each coder run is structured.
    # If the skill is generate_and_run, we print it.
    # If check_status returns SUCCESS or FIX, we print it.
    if skill == "generate_and_run":
        print(f"Idx {idx}: generate_run | Node {node_id} | Status {status}")
    elif skill == "check_status" and status in ["SUCCESS", "FIX"]:
        decision = item.get("output", {}).get("decision")
        score = item.get("output", {}).get("validation_score")
        err_class = item.get("output", {}).get("error_class") # wait, does error_class exist?
        err_summary = item.get("output", {}).get("error_summary")
        print(f"Idx {idx}: check_status | Node {node_id} | Status {status} | Decision {decision} | Score {score} | Err {err_summary[:50] if err_summary else None}")
