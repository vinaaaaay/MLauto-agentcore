import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

# Cost per 1M tokens
INPUT_COST_PER_1M = 0.435
OUTPUT_COST_PER_1M = 0.87

def parse_line_timestamp(line):
    parts = line.split(' | ', 1)
    if len(parts) == 2:
        try:
            # Handle 'Z' for UTC if present (for older Python versions)
            ts_str = parts[0].replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts_str)
            # astimezone(timezone.utc) converts naive datetimes using system timezone,
            # and aware datetimes using their parsed offset, aligning them both to UTC.
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze Bedrock AgentCore run metrics and output JSON.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run output directory (e.g. runs/run_19)")
    parser.add_argument("--out", type=str, default="metrics_summary.json", help="Output file name")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: run directory '{args.run_dir}' does not exist or is not a directory.")
        return

    cw_logs_path = run_dir / "cw_logs.txt"
    if not cw_logs_path.exists():
        print(f"Error: {cw_logs_path} not found.")
        return

    iteration_starts = {}
    benchmark_name = "unknown"
    run_id_name = run_dir.name
    e2e_workflow_s = 0.0

    iter_pattern = re.compile(r"NODE:\s+select_node\s+\(Iteration\s+(\d+)/")
    s3_pattern = re.compile(r"s3://mle-bench-lite/lite/([^/]+)/")
    e2e_pattern = re.compile(r"\[Orchestrator E2E\] Total time:\s*([\d\.]+)s")

    last_ts = None
    e2e_end_ts = None
    with open(cw_logs_path, "r", encoding="utf-8") as f:
        for line in f:
            ts = parse_line_timestamp(line)
            if ts:
                last_ts = ts
            
            m_iter = iter_pattern.search(line)
            if m_iter and ts:
                iter_num = int(m_iter.group(1))
                if iter_num not in iteration_starts:
                    iteration_starts[iter_num] = ts
                    
            m_s3 = s3_pattern.search(line)
            if m_s3:
                benchmark_name = m_s3.group(1)
                
            m_e2e = e2e_pattern.search(line)
            if m_e2e:
                e2e_workflow_s = float(m_e2e.group(1))
                if ts:
                    e2e_end_ts = ts

    # Fallback to client_metrics if needed
    client_metrics_path = run_dir / "client_metrics.json"
    if client_metrics_path.exists():
        try:
            with open(client_metrics_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                client_dur = c_data.get("workflow_duration ( client side )", 0.0)
                if client_dur > e2e_workflow_s:
                    e2e_workflow_s = client_dur
        except Exception:
            pass

    iterations = sorted(iteration_starts.keys())
    iteration_intervals = []
    for i in range(len(iterations)):
        start = iteration_starts[iterations[i]]
        if i + 1 < len(iterations):
            end = iteration_starts[iterations[i+1]]
        else:
            end = e2e_end_ts or last_ts or start
        iteration_intervals.append({
            "iteration": iterations[i],
            "start": start,
            "end": end
        })

    def get_iter_for_ts(ts):
        if not ts: return None
        for interval in iteration_intervals:
            if interval["iteration"] == 0 and ts < interval["start"]:
                return 0
            if interval["start"] <= ts <= interval["end"]:
                return interval["iteration"]
        return None

    totals = {
        "e2e_workflow_s": e2e_workflow_s,
        "tool_runtime_s": 0.0,
        "orchestrator_runtime_s": 0.0,
        "agent_runtime_s": {
            "perception": 0.0,
            "semantic": 0.0,
            "coder": 0.0,
            "total": 0.0
        },
        "llm_latency_s": 0.0,
        "infra_cost_usd": 0.0,
        "llm_cost_usd": 0.0,
        "total_cost_usd": 0.0
    }
    
    per_iter = {}
    total_other_iters_e2e = 0.0
    for interval in iteration_intervals:
        it = interval["iteration"]
        e2e_iter = (interval["end"] - interval["start"]).total_seconds()
        if it > 0:
            total_other_iters_e2e += e2e_iter
            
        per_iter[it] = {
            "iteration": it,
            "e2e_workflow_s": e2e_iter,
            "tool_runtime_s": 0.0,
            "orchestrator_runtime_s": 0.0,
            "agent_runtime_s": {"total": 0.0},
            "infra_cost_usd": 0.0,
            "llm_cost_usd": 0.0,
            "total_cost_usd": 0.0
        }
        
    per_iter[0]["e2e_workflow_s"] = e2e_workflow_s - total_other_iters_e2e

    seen_events = set()
    log_files = ["cw_logs.txt", "perception_cw_logs.txt", "semantic_cw_logs.txt", "coder_cw_logs.txt", "mcts_cw_logs.txt", "mcpserver_cw_logs.txt"]
    
    for log_file in log_files:
        path = run_dir / log_file
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ts = parse_line_timestamp(line)
                it = get_iter_for_ts(ts)
                
                idx = line.find('{')
                if idx != -1:
                    try:
                        data = json.loads(line[idx:])
                        event_type = data.get("event_type")
                        if not event_type:
                            continue
                            
                        if event_type == "psutil_metrics_node":
                            node_name = data.get("node_name")
                            ev_ts = data.get("timestamp")
                            e2e_s = data.get("node_e2e_s", 0.0)
                            
                            event_key = (event_type, log_file, node_name, ev_ts, e2e_s)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            
                            if node_name == "call_perception_agent":
                                totals["agent_runtime_s"]["perception"] += e2e_s
                                totals["agent_runtime_s"]["total"] += e2e_s
                                if it is not None:
                                    per_iter[it]["agent_runtime_s"]["total"] += e2e_s
                            elif node_name == "call_memory_agent":
                                totals["agent_runtime_s"]["semantic"] += e2e_s
                                totals["agent_runtime_s"]["total"] += e2e_s
                                if it is not None:
                                    per_iter[it]["agent_runtime_s"]["total"] += e2e_s
                            elif node_name == "call_coding_agent":
                                totals["agent_runtime_s"]["coder"] += e2e_s
                                totals["agent_runtime_s"]["total"] += e2e_s
                                if it is not None:
                                    per_iter[it]["agent_runtime_s"]["total"] += e2e_s
                                    
                        elif event_type == "tool_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("tool_call", log_file, run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            
                            latency_s = latency_ms / 1000.0
                            totals["tool_runtime_s"] += latency_s
                            if it is not None:
                                per_iter[it]["tool_runtime_s"] += latency_s
                                
                        elif event_type == "llm_call":
                            run_id = data.get("run_id") or data.get("timestamp")
                            latency_ms = data.get("latency_ms", 0.0)
                            event_key = ("llm_call", log_file, run_id, latency_ms)
                            if event_key in seen_events:
                                continue
                            seen_events.add(event_key)
                            
                            latency_s = data.get("wall_clock_s") or (latency_ms / 1000.0)
                            totals["llm_latency_s"] += latency_s
                            
                            inp_t = data.get("input_tokens", 0)
                            out_t = data.get("output_tokens", 0)
                            cache_t = data.get("cached_tokens", 0)
                            inp_non_cache = max(0, inp_t - cache_t)
                            
                            cost = (inp_non_cache / 1e6) * INPUT_COST_PER_1M + (out_t / 1e6) * OUTPUT_COST_PER_1M
                            totals["llm_cost_usd"] += cost
                            if it is not None:
                                per_iter[it]["llm_cost_usd"] += cost

                    except Exception:
                        pass

    totals["e2e_workflow_s"] = round(totals["e2e_workflow_s"], 2)
    totals["tool_runtime_s"] = round(totals["tool_runtime_s"], 2)
    totals["agent_runtime_s"]["perception"] = round(totals["agent_runtime_s"]["perception"], 2)
    totals["agent_runtime_s"]["semantic"] = round(totals["agent_runtime_s"]["semantic"], 2)
    totals["agent_runtime_s"]["coder"] = round(totals["agent_runtime_s"]["coder"], 2)
    totals["agent_runtime_s"]["total"] = round(totals["agent_runtime_s"]["total"], 2)
    
    # Orchestrator runtime is the total workflow time minus the time spent inside agent nodes
    totals["orchestrator_runtime_s"] = round(max(0.0, totals["e2e_workflow_s"] - totals["agent_runtime_s"]["total"]), 2)
    
    totals["llm_latency_s"] = round(totals["llm_latency_s"], 2)
    totals["llm_cost_usd"] = round(totals["llm_cost_usd"], 6)

    out_per_iter = []
    for it in sorted(per_iter.keys()):
        d = per_iter[it]
        d["e2e_workflow_s"] = round(d["e2e_workflow_s"], 2)
        d["tool_runtime_s"] = round(d["tool_runtime_s"], 2)
        d["agent_runtime_s"]["total"] = round(d["agent_runtime_s"]["total"], 2)
        
        # Orchestrator runtime for this specific iteration
        d["orchestrator_runtime_s"] = round(max(0.0, d["e2e_workflow_s"] - d["agent_runtime_s"]["total"]), 2)
        
        d["llm_cost_usd"] = round(d["llm_cost_usd"], 6)
        out_per_iter.append(d)

    final_output = {
        "run_id": run_id_name,
        "benchmark": benchmark_name,
        "totals": totals,
        "per_iteration": out_per_iter
    }

    out_path = run_dir / args.out
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
    print(f"Metrics successfully written to {out_path} (JSON format)")

    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()
