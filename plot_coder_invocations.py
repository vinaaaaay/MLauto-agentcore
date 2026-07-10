import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def parse_dt(ts_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            cleaned = ts_str.split("Z")[0].split("+")[0]
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unknown format: {ts_str}")

def parse_cw_dt(ts_str):
    for fmt in ("%Y-%m-%d__%H-%M-%S.%f", "%Y-%m-%d__%H-%M-%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unknown format: {ts_str}")

def classify_error(decision, score, err_sum, stderr):
    if decision == "SUCCESS":
        return "Successful Run"
    
    err_text = ((err_sum or "") + " " + (stderr or "")).lower()
    
    # Check for timeout (excluding standard AutoGluon headers)
    is_timeout = False
    if "did not complete within" in err_text or "timed out" in err_text or "killed" in err_text:
        is_timeout = True
    elif "timeout" in err_text and not ("time limit =" in err_text or "remaining time" in err_text):
        is_timeout = True
        
    if is_timeout:
        return "Timeout / Tool-Layer"
    elif "data selection" in err_text or "data_selection" in err_text:
        return "Data Selection"
    elif any(x in err_text for x in ["keyword argument", "unexpected keyword", "unsupported keyword", "not a valid keyword", "invalid keyword", "does not recognize"]):
        return "Invalid API Argument"
    else:
        return "Code Bug / Runtime Error"

def plot_coder_invocations(run_dir):
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        print(f"Error: Could not find {run_path}")
        sys.exit(1)

    # 1. Load coder_agent.json
    coder_json_path = run_path / "coder_agent.json"
    if not coder_json_path.exists():
        print(f"Error: Could not find {coder_json_path}")
        sys.exit(1)
        
    with open(coder_json_path, "r", encoding="utf-8") as f:
        coder_agent_data = json.load(f)

    # Group events by normalized node_id
    node_events = {}
    for item in coder_agent_data:
        node_id_val = item.get("input", {}).get("node_id") or item.get("input", {}).get("job_id")
        if node_id_val is None:
            continue
        
        # Normalize node_id to integer
        if isinstance(node_id_val, str) and node_id_val.startswith("node_"):
            node_id = int(node_id_val.split("_")[1])
        else:
            try:
                node_id = int(node_id_val)
            except ValueError:
                continue
        
        if node_id not in node_events:
            node_events[node_id] = []
        node_events[node_id].append(item)

    if not node_events:
        print("Error: No coder node invocations found in coder_agent.json")
        sys.exit(1)

    # Sort nodes by id
    sorted_node_ids = sorted(node_events.keys())
    nodes_info = {}
    
    for node_id in sorted_node_ids:
        items = node_events[node_id]
        items.sort(key=lambda x: x.get("timestamp", ""))
        
        first_item = items[0]
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
        
        start_dt = parse_dt(start_ts)
        end_dt = parse_dt(end_ts) + timedelta(seconds=time_taken)
        
        output = last_item.get("output", {})
        decision = output.get("decision", "FIX")
        validation_score = output.get("validation_score")
        error_summary = output.get("error_summary") or output.get("error")
        stderr = output.get("stderr", "")
        
        nodes_info[node_id] = {
            "start_dt": start_dt,
            "end_dt": end_dt,
            "total_duration": max(0.1, (end_dt - start_dt).total_seconds()),
            "decision": decision,
            "validation_score": validation_score,
            "error_summary": error_summary,
            "stderr": stderr,
            "llm_time": 0.0,
            "shell_time": 0.0,
            "other_tool_time": 0.0
        }

    # 2. Read coder_cw_logs.txt to get fine-grained timing
    cw_logs_path = run_path / "coder_cw_logs.txt"
    seen_events = set()
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
                    
                    event_dt = parse_cw_dt(ts_str)
                    
                    # Deduplicate events using run_id and a unique tuple fallback
                    run_id = data.get("run_id")
                    event_key = run_id if run_id else (data.get("timestamp"), data.get("latency_ms"), data.get("event_type"), data.get("node_name"))
                    if event_key in seen_events:
                        continue
                    seen_events.add(event_key)
                    
                    # Match event to node with 5-second padding
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
                except Exception:
                    pass

    # Post-process node times
    num_nodes = len(sorted_node_ids)
    col_labels = [str(node_id + 1) for node_id in sorted_node_ids]
    
    validation_scores = []
    decisions = []
    shell_exec_rtts = []
    llm_gens = []
    total_durations = []
    error_classes = []
    
    # Stacked bar datasets
    llm_plot_times = []
    shell_plot_times = []
    other_plot_times = []
    
    total_shell_rtt = 0.0
    
    for node_id in sorted_node_ids:
        info = nodes_info[node_id]
        
        # Values for Table
        score = info["validation_score"]
        validation_scores.append(f"{score:.4f}" if score is not None else "N/A")
        decisions.append(info["decision"])
        
        shell_time = info["shell_time"]
        total_shell_rtt += shell_time
        shell_exec_rtts.append(f"{shell_time:.1f}s")
        
        llm_time = info["llm_time"]
        llm_gens.append(f"{llm_time:.1f}s")
        
        tot_dur = info["total_duration"]
        total_durations.append(f"{tot_dur:.1f}s")
        
        err_class = classify_error(info["decision"], info["validation_score"], info["error_summary"], info["stderr"])
        error_classes.append(err_class)
        
        # Values for Stacked Bar Chart
        llm_plot_times.append(llm_time)
        shell_plot_times.append(shell_time)
        
        # Other Tool Calls is the remaining time of the actual total duration
        other_time = max(0.0, tot_dur - llm_time - shell_time)
        other_plot_times.append(other_time)

    # 3. Plotting
    x = np.arange(num_nodes)
    width = 0.55

    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Draw stacked bars
    # Bottom layer: LLM Generation (Red)
    p1 = ax.bar(x, llm_plot_times, width, label='Cumulative LLM Generation (All Nodes)', color='#E55B3C')
    # Middle layer: Other Tool Calls (Grey)
    p2 = ax.bar(x, other_plot_times, width, bottom=llm_plot_times, label='Other Tool Calls', color='#B0BEC5')
    # Top layer: Shell Exec RTT (Blue)
    bottom_for_shell = [l + o for l, o in zip(llm_plot_times, other_plot_times)]
    p3 = ax.bar(x, shell_plot_times, width, bottom=bottom_for_shell, label='Shell Exec RTT (Sandbox Wall-Clock)', color='#3498DB')

    # Styling axes
    ax.set_ylabel('Duration (Seconds)', fontsize=12, fontweight='bold', color='#2C3E50', labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # Hide standard x-axis labels/ticks
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    # Set y limits with padding
    max_tot = max([info["total_duration"] for info in nodes_info.values()])
    ax.set_ylim(0, max_tot * 1.15)
    
    # Add text labels on top of bars
    for i in range(num_nodes):
        tot_dur = nodes_info[sorted_node_ids[i]]["total_duration"]
        ax.text(i, tot_dur + (max_tot * 0.015), f"{tot_dur:.1f}s",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C3E50')

    # Add Legend above the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=10)

    # Add Total Shell Exec RTT text
    ax.text(0.98, 0.95, f"Total Shell Exec RTT: {total_shell_rtt:.1f}s",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold', color='#2C3E50')

    # Construct and Style the Table
    cell_text = [
        validation_scores,
        decisions,
        shell_exec_rtts,
        llm_gens,
        total_durations,
        error_classes
    ]
    row_labels = [
        "Validation Score",
        "Decision",
        "Shell Exec RTT",
        "Cumulative LLM Gen",
        "Total Duration",
        "Error Class"
    ]
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='bottom',
        bbox=[0, -0.45, 1, 0.35]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Custom styling for table cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)
        
        # Header cells (column labels and row labels)
        if row == 0 or col == -1:
            cell.set_facecolor('#1C3B57')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.get_text().set_horizontalalignment('center')
        else:
            # Data cells
            cell.set_facecolor('#FFFFFF')
            cell.get_text().set_color('#2C3E50')
            cell.get_text().set_horizontalalignment('center')
            
            # Highlight Validation Score
            if row == 1:
                val_text = cell.get_text().get_text()
                if val_text != "N/A":
                    cell.get_text().set_color('#27AE60')
                    cell.get_text().set_weight('bold')
            
            # Highlight Decision
            if row == 2:
                dec_text = cell.get_text().get_text()
                if dec_text == "SUCCESS":
                    cell.get_text().set_color('#27AE60')
                    cell.get_text().set_weight('bold')
                elif dec_text == "FIX":
                    cell.get_text().set_color('#C0392B')
                    cell.get_text().set_weight('bold')

    # Adjust layout to fit table
    plt.subplots_adjust(bottom=0.32, top=0.88, left=0.15, right=0.98)
    
    out_path = run_path / "coder_invocations_bar.png"
    plt.savefig(out_path, dpi=150, facecolor='#FFFFFF', edgecolor='none')
    plt.close()
    
    print(f"\n✔ Custom bar chart with summary table saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_coder_invocations.py <run_dir>")
        sys.exit(1)
        
    run_dir = sys.argv[1]
    plot_coder_invocations(run_dir)
