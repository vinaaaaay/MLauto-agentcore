import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import timedelta

# Import helper functions from analyze_run_metrics
from analyze_run_metrics import parse_timestamps, get_agent_info, fetch_metrics

def plot_coder_invocations(run_dir):
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        print(f"Error: Could not find {run_path}")
        sys.exit(1)

    # 1. Parse run timestamps
    try:
        start_dt, end_dt = parse_timestamps(run_path)
    except Exception as e:
        print(f"Error parsing timestamps from logs.txt: {e}")
        sys.exit(1)

    # 2. Parse Orchestrator call_coding_agent E2E times from local logs.txt
    logs_file = run_path / "logs.txt"
    orchestrator_node_e2e = []
    
    if logs_file.exists():
        with open(logs_file, "r", encoding="utf-8") as f:
            for line in f:
                if "psutil_metrics_node" in line and '"node_name": "call_coding_agent"' in line:
                    idx = line.find('{')
                    if idx != -1:
                        try:
                            data = json.loads(line[idx:])
                            orchestrator_node_e2e.append(data.get("node_e2e_s", 0.0))
                        except Exception:
                            pass
                            
    if not orchestrator_node_e2e:
        print("Error: Could not parse any call_coding_agent node events from logs.txt.")
        sys.exit(1)

    print(f"Found {len(orchestrator_node_e2e)} orchestrator iterations.")

    # 3. Query CloudWatch for Coder Agent Graph metrics
    start_time = (start_dt - timedelta(minutes=1)).timestamp()
    end_time = (end_dt + timedelta(minutes=2)).timestamp()

    root_dir = run_path.parent.parent
    coder_dir = root_dir / "coder_agent"
    
    info = get_agent_info(coder_dir)
    if not info:
        print("Error: Could not find Coder Agent configuration to query CloudWatch.")
        sys.exit(1)

    print("Fetching Coder Agent metrics from CloudWatch...")
    all_metrics = []
    for name, arn in info:
        agent_hash = arn.split("/")[-1]
        log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_hash}-DEFAULT"
        records = fetch_metrics(log_group_name, "coder_agent", start_time, end_time)
        all_metrics.extend(records)

    # Deduplicate and filter Coder Graph events
    coder_graphs = []
    seen_keys = set()
    for record in all_metrics:
        if record.get("event_type") == "psutil_metrics_graph":
            timestamp = record.get("timestamp") or record.get("_timestamp")
            e2e = record.get("graph_e2e_s", 0.0)
            key = (timestamp, round(e2e, 4))
            if key not in seen_keys:
                seen_keys.add(key)
                coder_graphs.append(record)

    # Sort chronologically
    coder_graphs.sort(key=lambda x: x.get("timestamp", ""))

    if len(coder_graphs) != len(orchestrator_node_e2e):
        print(f"Warning: Mismatch between orchestrator nodes ({len(orchestrator_node_e2e)}) and Coder Graph events ({len(coder_graphs)}). Using minimum count.")
    
    num_nodes = min(len(orchestrator_node_e2e), len(coder_graphs))

    # Calculate generation and training times
    labels = [f"Node {i}" for i in range(num_nodes)]
    generation_times = []
    training_times = []
    total_times = []

    for i in range(num_nodes):
        total_e2e = orchestrator_node_e2e[i]
        gen_time = coder_graphs[i].get("graph_e2e_s", 0.0)
        train_time = max(0.0, total_e2e - gen_time)
        
        generation_times.append(gen_time)
        training_times.append(train_time)
        total_times.append(total_e2e)

    # Plotting
    x = np.arange(num_nodes)
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8F9F9')

    # Stacked bars: Generation time on bottom, Training time on top
    p1 = ax.bar(x, generation_times, width, label='Code Generation (LLM Calls)', color='#2E86C1')
    p2 = ax.bar(x, training_times, width, bottom=generation_times, label='Model Training (Sandbox Background)', color='#E74C3C')

    ax.set_ylabel('Execution Time (Seconds)', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_title(f'Actual MCTS Node Execution & Training Times - {run_path.name}', fontsize=15, fontweight='bold', pad=20, color='#2C3E50')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    
    ax.legend(loc='upper right', fontsize=12, facecolor='#FFFFFF', edgecolor='#BDC3C7')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of bars
    for i in range(num_nodes):
        total = total_times[i]
        ax.text(x[i], total + (max(total_times) * 0.01), f"{int(total)}s", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#34495E')

    plt.tight_layout()

    out_path = run_path / "coder_invocations_bar.png"
    plt.savefig(out_path, dpi=150, facecolor='#FFFFFF', edgecolor='none')
    plt.close()

    print(f"\n✔ Successfully saved per-node training execution bar chart to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_coder_invocations.py <run_dir>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    plot_coder_invocations(run_dir)
