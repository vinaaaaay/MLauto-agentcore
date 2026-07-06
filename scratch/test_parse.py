import json
import re
from datetime import datetime, timedelta
from pathlib import Path

run_dir = Path("/home/administrator/dreamlab/MLauto-agentcore/runs/run_21")

# Load coder_agent.json
with open(run_dir / "coder_agent.json", "r") as f:
    coder_agent_data = json.load(f)

# Find all events in coder_agent.json grouped by node_id
node_events = {}
for item in coder_agent_data:
    node_id_val = item.get("input", {}).get("node_id") or item.get("input", {}).get("job_id")
    if node_id_val is None:
        continue
    # Normalize node_id_val to integer
    if isinstance(node_id_val, str) and node_id_val.startswith("node_"):
        node_id = int(node_id_val.split("_")[1])
    else:
        node_id = int(node_id_val)
    
    if node_id not in node_events:
        node_events[node_id] = []
    node_events[node_id].append(item)

nodes_info = {}
for node_id, items in node_events.items():
    # Sort by timestamp
    items.sort(key=lambda x: x.get("timestamp", ""))
    
    first_item = items[0]
    # Find last check_status with COMPLETED/FAILED
    last_item = None
    for item in reversed(items):
        if item.get("skill") == "check_status" and item.get("output", {}).get("status") in ["COMPLETED", "FAILED"]:
            last_item = item
            break
    
    if not last_item:
        last_item = items[-1]
        
    start_ts = first_item["timestamp"]
    end_ts = last_item["timestamp"]
    time_taken = last_item.get("time_taken_seconds", 0.0)
    
    start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00")) + timedelta(seconds=time_taken)
    
    output = last_item.get("output", {})
    decision = output.get("decision", "FIX")
    validation_score = output.get("validation_score")
    error_summary = output.get("error_summary")
    stderr = output.get("stderr", "")
    
    nodes_info[node_id] = {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "total_duration": (end_dt - start_dt).total_seconds(),
        "decision": decision,
        "validation_score": validation_score,
        "error_summary": error_summary,
        "stderr": stderr,
        "llm_time": 0.0,
        "shell_time": 0.0,
        "other_tool_time": 0.0
    }

# Read coder_cw_logs.txt
cw_logs_path = run_dir / "coder_cw_logs.txt"
if cw_logs_path.exists():
    with open(cw_logs_path, "r", encoding="utf-8") as f:
        for line in f:
            idx = line.find("{")
            if idx == -1:
                continue
            try:
                data = json.loads(line[idx:])
                ts_str = data.get("timestamp")
                if not ts_str:
                    continue
                # Format: 2026-07-05__21-43-29.106833
                event_dt = datetime.strptime(ts_str, "%Y-%m-%d__%H-%M-%S.%f")
                
                # Find matching node with padding
                matched_node_id = None
                for node_id, info in nodes_info.items():
                    if info["start_dt"] - timedelta(seconds=5) <= event_dt <= info["end_dt"] + timedelta(seconds=5):
                        matched_node_id = node_id
                        break
                
                if matched_node_id is not None:
                    event_type = data.get("event_type")
                    latency_ms = data.get("latency_ms") or 0.0
                    wall_clock_s = data.get("wall_clock_s")
                    duration = wall_clock_s if wall_clock_s is not None else (latency_ms / 1000.0)
                    
                    if event_type == "llm_call":
                        nodes_info[matched_node_id]["llm_time"] += duration
                    elif event_type == "tool_call":
                        tool_name = data.get("tool_name")
                        if tool_name == "sandbox_exec_shell":
                            nodes_info[matched_node_id]["shell_time"] += duration
                        else:
                            nodes_info[matched_node_id]["other_tool_time"] += duration
            except Exception as e:
                pass

def classify_error(decision, score, err_sum, stderr):
    if decision == "SUCCESS":
        return "Successful Run"
    
    err_text = ((err_sum or "") + " " + (stderr or "")).lower()
    
    if "time limit" in err_text or "timeout" in err_text or "timed out" in err_text or "killed" in err_text:
        return "Timeout / Tool-Layer"
    elif "data selection" in err_text or "data_selection" in err_text:
        return "Data Selection"
    else:
        return "Invalid API Argument"

print("Node parsing results:")
for node_id in sorted(nodes_info.keys()):
    info = nodes_info[node_id]
    err_class = classify_error(info["decision"], info["validation_score"], info["error_summary"], info["stderr"])
    print(f"Node {node_id}:")
    print(f"  Total Duration: {info['total_duration']:.2f}s")
    print(f"  LLM Gen Time:   {info['llm_time']:.2f}s")
    print(f"  Shell Exec:     {info['shell_time']:.2f}s")
    print(f"  Other Tools:    {info['total_duration'] - info['llm_time'] - info['shell_time']:.2f}s")
    print(f"  Decision:       {info['decision']}")
    print(f"  Score:          {info['validation_score']}")
    print(f"  Error Class:    {err_class}")
    print()
