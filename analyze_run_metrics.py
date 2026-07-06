import argparse
import json
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze Bedrock AgentCore run metrics.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run output directory (e.g. runs/run_19)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: run directory '{args.run_dir}' does not exist or is not a directory.")
        return

    cw_logs_path = run_dir / "cw_logs.txt"
    if not cw_logs_path.exists():
        print(f"Error: {cw_logs_path} not found.")
        return

    # Regex to extract e2e time and peak ram
    e2e_pattern = re.compile(r"\[Orchestrator E2E\] Total time: ([\d\.]+)s \| Peak RAM: ([\d\.]+) MB")
    
    orch_e2e_duration = 0.0
    peak_ram_gb = 0.0
    sync_s3 = 0.0
    coder_duration = 0.0
    perception_duration = 0.0
    semantic_duration = 0.0
    mcts_duration = 0.0
    seen_events = set()

    with open(cw_logs_path, "r", encoding="utf-8") as f:
        for line in f:
            match = e2e_pattern.search(line)
            if match:
                orch_e2e_duration = float(match.group(1))
                peak_ram_mb = float(match.group(2))
                peak_ram_gb = round(peak_ram_mb / 1024.0, 4)
                
            idx = line.find('{')
            if idx != -1:
                try:
                    data = json.loads(line[idx:])
                    event_type = data.get("event_type")
                    if event_type == "psutil_metrics_node":
                        node_name = data.get("node_name")
                        timestamp = data.get("timestamp")
                        e2e_s = data.get("node_e2e_s", 0.0)
                        
                        # Filter duplicates using unique tuple
                        event_key = (event_type, node_name, timestamp, e2e_s)
                        if event_key in seen_events:
                            continue
                        seen_events.add(event_key)
                        
                        if node_name == "sync_s3_to_sandbox":
                            sync_s3 = round(e2e_s, 4)
                        elif node_name == "call_coding_agent":
                            coder_duration += e2e_s
                        elif node_name == "call_perception_agent":
                            perception_duration += e2e_s
                        elif node_name == "call_memory_agent":
                            semantic_duration += e2e_s
                        elif node_name in ["init_mcts", "select_node", "expand_node", "update_node", "backpropagate", "finalize_results"]:
                            mcts_duration += e2e_s
                except Exception:
                    pass

    coder_duration = round(coder_duration, 4)
    perception_duration = round(perception_duration, 4)
    semantic_duration = round(semantic_duration, 4)
    mcts_duration = round(mcts_duration, 4)

    # Parse coder LLM & tool metrics
    coder_cw_logs_path = run_dir / "coder_cw_logs.txt"
    coder_llm_latency = 0.0
    coder_tool_duration = 0.0
    coder_peak_ram = 0.0
    coder_input_tokens = 0
    coder_cached_tokens = 0
    coder_output_tokens = 0
    coder_reasoning_tokens = 0

    if coder_cw_logs_path.exists():
        with open(coder_cw_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        event_type = data.get("event_type")
                        if event_type == "llm_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = (event_type, run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            
                            coder_llm_latency += data.get("wall_clock_s") or (latency_ms / 1000.0)
                            coder_input_tokens += data.get("input_tokens", 0)
                            coder_cached_tokens += data.get("cached_tokens", 0)
                            coder_output_tokens += data.get("output_tokens", 0)
                            coder_reasoning_tokens += data.get("reasoning_tokens", 0)
                        elif event_type == "psutil_metrics_node":
                            ram = data.get("peak_RAM_GB")
                            if ram is not None:
                                coder_peak_ram = max(coder_peak_ram, float(ram))
                        elif event_type == "tool_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("coder_tool", run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            coder_tool_duration += latency_ms / 1000.0
                    except Exception:
                        pass

    coder_llm_latency = round(coder_llm_latency, 4)
    coder_tool_duration = round(coder_tool_duration, 4)
    coder_input_tokens_non_cached = max(0, coder_input_tokens - coder_cached_tokens)
    
    # Calculate coder costs
    input_cost = round((coder_input_tokens_non_cached / 1000000.0) * 0.435, 8)
    cached_cost = 0.0
    output_cost = round((coder_output_tokens / 1000000.0) * 0.87, 8)
    llm_total_cost = round(input_cost + cached_cost + output_cost, 8)

    # -------------------------------------------------------------------------
    # Parse perception metrics
    # -------------------------------------------------------------------------
    # E2E duration: from orchestrator cw_logs.txt (call_perception_agent node)
    # E2E duration was already parsed in the first loop over cw_logs.txt

    # LLM latency & token breakdown: from perception_cw_logs.txt
    perception_cw_logs_path = run_dir / "perception_cw_logs.txt"
    perception_llm_latency = 0.0
    perception_tool_duration = 0.0
    perception_peak_ram = 0.0
    perception_input_tokens = 0
    perception_cached_tokens = 0
    perception_output_tokens = 0
    perception_reasoning_tokens = 0

    if perception_cw_logs_path.exists():
        with open(perception_cw_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        event_type = data.get("event_type")
                        if event_type == "llm_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("perception_llm", run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            perception_llm_latency += data.get("wall_clock_s") or (latency_ms / 1000.0)
                            perception_input_tokens += data.get("input_tokens", 0)
                            perception_cached_tokens += data.get("cached_tokens", 0)
                            perception_output_tokens += data.get("output_tokens", 0)
                            perception_reasoning_tokens += data.get("reasoning_tokens", 0)
                        elif event_type == "tool_call":
                            node_name = data.get("node_name")
                            if node_name in ["scan_data", "find_description_files"]:
                                run_id = data.get("run_id") or data.get("timestamp")
                                latency_ms = data.get("latency_ms", 0.0)
                                event_key = ("perception_tool", run_id, latency_ms)
                                if event_key in seen_events:
                                    continue
                                seen_events.add(event_key)
                                perception_tool_duration += latency_ms / 1000.0
                        elif event_type == "psutil_metrics_node":
                            ram = data.get("peak_RAM_GB")
                            if ram is not None:
                                perception_peak_ram = max(perception_peak_ram, float(ram))
                    except Exception:
                        pass

    perception_llm_latency = round(perception_llm_latency, 4)
    perception_tool_duration = round(perception_tool_duration, 4)
    perception_input_tokens_non_cached = max(0, perception_input_tokens - perception_cached_tokens)
    perc_input_cost = round((perception_input_tokens_non_cached / 1000000.0) * 0.435, 8)
    perc_cached_cost = 0.0
    perc_output_cost = round((perception_output_tokens / 1000000.0) * 0.87, 8)
    perc_llm_total_cost = round(perc_input_cost + perc_cached_cost + perc_output_cost, 8)

    # -------------------------------------------------------------------------
    # Parse semantic metrics
    # -------------------------------------------------------------------------
    # E2E duration: from orchestrator cw_logs.txt (call_memory_agent node)
    # LLM latency & token breakdown: from semantic_cw_logs.txt
    semantic_cw_logs_path = run_dir / "semantic_cw_logs.txt"
    semantic_llm_latency = 0.0
    semantic_tool_duration = 0.0
    semantic_peak_ram = 0.0
    semantic_input_tokens = 0
    semantic_cached_tokens = 0
    semantic_output_tokens = 0
    semantic_reasoning_tokens = 0

    if semantic_cw_logs_path.exists():
        with open(semantic_cw_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        event_type = data.get("event_type")
                        if event_type == "llm_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("semantic_llm", run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            semantic_llm_latency += data.get("wall_clock_s") or (latency_ms / 1000.0)
                            semantic_input_tokens += data.get("input_tokens", 0)
                            semantic_cached_tokens += data.get("cached_tokens", 0)
                            semantic_output_tokens += data.get("output_tokens", 0)
                            semantic_reasoning_tokens += data.get("reasoning_tokens", 0)
                        elif event_type == "tool_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("semantic_tool", run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            semantic_tool_duration += latency_ms / 1000.0
                        elif event_type == "psutil_metrics_node":
                            ram = data.get("peak_RAM_GB")
                            if ram is not None:
                                semantic_peak_ram = max(semantic_peak_ram, float(ram))
                    except Exception:
                        pass

    semantic_llm_latency = round(semantic_llm_latency, 4)
    semantic_tool_duration = round(semantic_tool_duration, 4)
    semantic_input_tokens_non_cached = max(0, semantic_input_tokens - semantic_cached_tokens)
    sem_input_cost = round((semantic_input_tokens_non_cached / 1000000.0) * 0.435, 8)
    sem_cached_cost = 0.0
    sem_output_cost = round((semantic_output_tokens / 1000000.0) * 0.87, 8)
    sem_llm_total_cost = round(sem_input_cost + sem_cached_cost + sem_output_cost, 8)

    # -------------------------------------------------------------------------
    # Parse MCTS metrics
    # -------------------------------------------------------------------------
    mcts_cw_logs_path = run_dir / "mcts_cw_logs.txt"
    mcts_peak_ram = 0.0

    if mcts_cw_logs_path.exists():
        with open(mcts_cw_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        ram = data.get("peak_RAM_GB")
                        if ram is not None:
                            mcts_peak_ram = max(mcts_peak_ram, float(ram))
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Parse MCP Server metrics
    # -------------------------------------------------------------------------
    mcpserver_cw_logs_path = run_dir / "mcpserver_cw_logs.txt"
    mcpserver_peak_ram = 0.0

    if mcpserver_cw_logs_path.exists():
        with open(mcpserver_cw_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        ram = data.get("peak_RAM_GB")
                        if ram is not None:
                            mcpserver_peak_ram = max(mcpserver_peak_ram, float(ram))
                    except Exception:
                        pass

    # Try parsing client-side metrics if available
    workflow_duration = 0.0
    client_metrics_path = run_dir / "client_metrics.json"
    if client_metrics_path.exists():
        try:
            with open(client_metrics_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                workflow_duration = c_data.get("workflow_duration ( client side )", 0.0)
        except Exception as e:
            print(f"Warning: Failed to parse client metrics: {e}")

    # Build the output JSON structure iteratively
    output = {
        "orch_latency (s)": {
            "workflow_duration ( client side )": workflow_duration,
            "orch_e2e_duration": orch_e2e_duration,
            "sync_s3": sync_s3
        },
        "orch_cost ($)": {
            "peak_ram_gb": peak_ram_gb
        },
        "perception": {
            "agent_latency (s)": {
                "total_e2e_duration": perception_duration,
                "llm_latency": perception_llm_latency,
                "total_tool_call_duration": perception_tool_duration
            },
            "cost ($)": {
                "peak_ram_gb": perception_peak_ram,
                "llm_total_cost": perc_llm_total_cost,
                "llm_breakdown": {
                    "input_tokens": perception_input_tokens,
                    "input_tokens_non_cached": perception_input_tokens_non_cached,
                    "cached_tokens": perception_cached_tokens,
                    "output_tokens": perception_output_tokens,
                    "reasoning_tokens": perception_reasoning_tokens,
                    "input_cost": perc_input_cost,
                    "cached_cost": perc_cached_cost,
                    "output_cost": perc_output_cost
                }
            }
        },
        "semantic": {
            "agent_latency (s)": {
                "total_e2e_duration": semantic_duration,
                "llm_latency": semantic_llm_latency,
                "total_tool_call_duration": semantic_tool_duration
            },
            "cost ($)": {
                "peak_ram_gb": semantic_peak_ram,
                "llm_total_cost": sem_llm_total_cost,
                "llm_breakdown": {
                    "input_tokens": semantic_input_tokens,
                    "input_tokens_non_cached": semantic_input_tokens_non_cached,
                    "cached_tokens": semantic_cached_tokens,
                    "output_tokens": semantic_output_tokens,
                    "reasoning_tokens": semantic_reasoning_tokens,
                    "input_cost": sem_input_cost,
                    "cached_cost": sem_cached_cost,
                    "output_cost": sem_output_cost
                }
            }
        },
        "mcts": {
            "agent_latency (s)": {
                "total_e2e_duration": mcts_duration
            },
            "cost ($)": {
                "peak_ram_gb": mcts_peak_ram
            }
        },
        "mcpserver": {
            "cost ($)": {
                "peak_ram_gb": mcpserver_peak_ram
            }
        },
        "coder": {
            "agent_latency (s)": {
                "total_e2e_duration": coder_duration,
                "llm_latency": coder_llm_latency,
                "total_tool_call_duration": coder_tool_duration
            },
            "cost ($)": {
                "peak_ram_gb": coder_peak_ram,
                "llm_total_cost": llm_total_cost,
                "llm_breakdown": {
                    "input_tokens": coder_input_tokens,
                    "input_tokens_non_cached": coder_input_tokens_non_cached,
                    "cached_tokens": coder_cached_tokens,
                    "output_tokens": coder_output_tokens,
                    "reasoning_tokens": coder_reasoning_tokens,
                    "input_cost": input_cost,
                    "cached_cost": cached_cost,
                    "output_cost": output_cost
                }
            }
        }
    }

    out_file = run_dir / "latency_and_cost.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Metrics successfully written to {out_file}")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
