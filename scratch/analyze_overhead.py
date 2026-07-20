import json
import re
from pathlib import Path

def analyze_orchestrator_overhead(run_dir_str):
    run_dir = Path(run_dir_str)
    cw_logs = run_dir / "cw_logs.txt"
    
    nodes = {}
    total_node_time = 0.0
    
    # We will also track the first timestamp and last timestamp
    start_ts = None
    end_ts = None
    
    e2e_pattern = re.compile(r"\[Orchestrator E2E\] Total time:\s*([\d\.]+)s")
    orch_e2e = 0.0
    
    with open(cw_logs, "r") as f:
        for line in f:
            # find E2E
            m = e2e_pattern.search(line)
            if m:
                orch_e2e = float(m.group(1))
                
            idx = line.find('{')
            if idx != -1:
                try:
                    data = json.loads(line[idx:])
                    if data.get("event_type") == "psutil_metrics_node":
                        name = data.get("node_name")
                        dur = data.get("node_e2e_s", 0.0)
                        nodes[name] = nodes.get(name, 0.0) + dur
                        total_node_time += dur
                except:
                    pass

    print(f"--- Analysis for {run_dir_str} ---")
    print(f"Total Orchestrator E2E: {orch_e2e:.4f}s")
    print(f"Sum of ALL node_e2e_s:  {total_node_time:.4f}s")
    print(f"Unaccounted Gap:        {orch_e2e - total_node_time:.4f}s")
    print("\nNode Breakdown:")
    for k, v in sorted(nodes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}s")
    print("="*40)

if __name__ == "__main__":
    analyze_orchestrator_overhead("runs/run_22")
    analyze_orchestrator_overhead("runs/run_28")
