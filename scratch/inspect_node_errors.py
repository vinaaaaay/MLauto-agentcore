import json
from pathlib import Path

run_dir = Path("/home/administrator/dreamlab/MLauto-agentcore/runs/run_21")
coder_json_path = run_dir / "coder_agent.json"

with open(coder_json_path, "r") as f:
    data = json.load(f)

# Find all completed nodes
completed_nodes = {}
for item in data:
    skill = item.get("skill")
    status = item.get("output", {}).get("status")
    if skill == "check_status" and status in ["COMPLETED", "FAILED"]:
        node_id = item.get("input", {}).get("job_id")
        # Keep the last completion for each node_id
        completed_nodes[node_id] = item

for node_id in sorted(completed_nodes.keys(), key=lambda x: int(x.split("_")[1])):
    item = completed_nodes[node_id]
    out = item.get("output", {})
    decision = out.get("decision")
    score = out.get("validation_score")
    err_sum = out.get("error_summary")
    stderr = out.get("stderr", "")
    
    print(f"--- {node_id} ---")
    print(f"Decision: {decision}")
    print(f"Score: {score}")
    print(f"Error Summary: {err_sum}")
    print(f"Stderr (first 200 chars): {stderr[:200]}")
    print()
