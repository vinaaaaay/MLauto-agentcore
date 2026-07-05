import os
import sys
import json
import yaml
import argparse
import re
import boto3
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Pricing
LLM_INPUT_COST_PER_1M = 0.435
LLM_OUTPUT_COST_PER_1M = 0.87

def get_agent_info(directory: Path):
    yaml_path = directory / ".bedrock_agentcore.yaml"
    if not yaml_path.exists():
        return None
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        agents = config.get("agents", {})
        info = []
        for name, details in agents.items():
            arn = details.get("bedrock_agentcore", {}).get("agent_arn")
            if arn:
                info.append((name, arn))
        return info
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        return None

def parse_timestamps(run_dir: Path):
    logs_file = run_dir / "logs.txt"
    if not logs_file.exists():
        raise FileNotFoundError(f"logs.txt not found in {run_dir}")
        
    start_dt = None
    end_dt = None
    
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    
    with open(logs_file, "r", encoding="utf-8") as f:
        for line in f:
            match = ts_pattern.match(line)
            if match:
                dt_str = match.group(1)
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                if start_dt is None:
                    start_dt = dt
                end_dt = dt
                
    if start_dt is None or end_dt is None:
        raise ValueError(f"Could not parse start or end timestamps in {logs_file}")
        
    return start_dt, end_dt

def fetch_metrics(log_group_name, service_name, start_time_epoch, end_time_epoch):
    logs_client = boto3.client("logs", region_name="ap-south-1")
    metrics_records = []
    
    try:
        paginator = logs_client.get_paginator('filter_log_events')
        # We REMOVED the filterPattern to pull llm_call and tool_call along with psutil_metrics
        pages = paginator.paginate(
            logGroupName=log_group_name,
            startTime=int(start_time_epoch * 1000),
            endTime=int(end_time_epoch * 1000)
        )
        
        for page in pages:
            events = page.get("events", [])
            for event in events:
                msg = event['message'].strip()
                try:
                    idx = msg.find('{')
                    if idx != -1:
                        data = json.loads(msg[idx:])
                        if "event_type" in data:
                            data["_service"] = service_name
                            data["_timestamp"] = event['timestamp']
                            metrics_records.append(data)
                except Exception:
                    pass
    except Exception as e:
        pass
        
    return metrics_records

def main():
    parser = argparse.ArgumentParser(description="Analyze Bedrock AgentCore run metrics.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run output directory (e.g. runs/run_12)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: run directory '{args.run_dir}' does not exist or is not a directory.")
        sys.exit(1)
        
    try:
        start_dt, end_dt = parse_timestamps(run_dir)
        workflow_duration = (end_dt - start_dt).total_seconds()
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        sys.exit(1)
        
    # Query with a buffer of 1 min before and 2 min after
    start_time = (start_dt - timedelta(minutes=1)).timestamp()
    end_time = (end_dt + timedelta(minutes=2)).timestamp()
    
    root_dir = run_dir.parent.parent
    services = [
        "MLorchestrator",
        "coder_agent",
        "perception_agent",
        "semantic_agent",
        "mcts_handler",
        "mcpserver"
    ]
    
    all_metrics = []
    
    # Query CloudWatch
    for service in services:
        dir_path = root_dir / service
        info = get_agent_info(dir_path)
        if info:
            for name, arn in info:
                agent_hash = arn.split("/")[-1]
                log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_hash}-DEFAULT"
                records = fetch_metrics(log_group_name, service, start_time, end_time)
                all_metrics.extend(records)
                
    # Parse from local logs.txt
    local_logs_path = run_dir / "logs.txt"
    if local_logs_path.exists():
        with open(local_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                if "psutil_metrics" in line:
                    try:
                        idx = line.find('{')
                        if idx != -1:
                            data = json.loads(line[idx:])
                            data["_service"] = "MLorchestrator"
                            data["_timestamp"] = int(start_time * 1000)
                            all_metrics.append(data)
                    except Exception:
                        pass

    # Deduplicate 
    unique_metrics = []
    seen_keys = set()
    for record in all_metrics:
        event_type = record.get("event_type")
        timestamp = record.get("timestamp") or record.get("_timestamp")
        
        if event_type in ["psutil_metrics_graph", "psutil_metrics_node"]:
            name = record.get("node_name") or record.get("graph_name") or "unknown"
            e2e = record.get("graph_e2e_s") or record.get("node_e2e_s") or 0.0
            key = (event_type, name, timestamp, round(e2e, 4))
        elif event_type == "llm_call":
            key = (event_type, record.get("run_id", timestamp))
        elif event_type == "tool_call":
            key = (event_type, record.get("run_id", timestamp))
        else:
            continue
            
        if key not in seen_keys:
            seen_keys.add(key)
            unique_metrics.append(record)
            
    all_metrics = unique_metrics

    # Process Metrics
    orch_latency = {
        "workflow_duration ( client side )": round(workflow_duration, 4),
        "orch_runtime_duration": 0.0,
        "agent_runtime_duration": 0.0,
        "sync_s3": 0.0,
    }
    
    orch_cost = {
        "billed_duration": 0.0,
        "sync_s3_billed_duration": 0.0,
        "peak_ram_gb": 0.0
    }
    
    agents = {}

    for record in all_metrics:
        service = record["_service"]
        evt = record["event_type"]
        
        # We include mcpserver and mcts_handler as well since the user requested it
        is_subagent = service in ["coder_agent", "perception_agent", "semantic_agent", "mcpserver", "mcts_handler"]
        if is_subagent and service not in agents:
            agents[service] = {
                "agent_latency (s)": {
                    "duration": 0.0,
                    "runtime_duration": 0.0,
                    "llm_latency": 0.0,
                    "total_tool_call_duration": 0.0
                },
                "cost ($)": {
                    "billed_duration": 0.0,
                    "peak_ram_gb": 0.0,
                    "llm_total_cost": 0.0,
                    "llm_breakdown": {
                        "input_tokens": 0,
                        "input_tokens_non_cached": 0,
                        "cached_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "input_cost": 0.0,
                        "cached_cost": 0.0,
                        "output_cost": 0.0
                    }
                }
            }
            
        # Update Peak RAM for ALL events
        peak_ram = record.get("peak_RAM_GB") or record.get("peak_ram_gb") or 0.0
        if service == "MLorchestrator":
            orch_cost["peak_ram_gb"] = max(orch_cost["peak_ram_gb"], peak_ram)
        elif is_subagent:
            agents[service]["cost ($)"]["peak_ram_gb"] = max(agents[service]["cost ($)"]["peak_ram_gb"], peak_ram)
            
        if evt == "psutil_metrics_graph":
            if service == "MLorchestrator":
                # Active time for orchestrator is e2e_s - wait_s
                active_time = record.get("graph_e2e_s", 0.0) - record.get("wait_time_s", 0.0)
                orch_latency["orch_runtime_duration"] += active_time
                orch_cost["billed_duration"] += active_time # Billed only for active compute in AgentCore
            elif is_subagent:
                e2e = record.get("graph_e2e_s", 0.0)
                active_time = e2e - record.get("wait_time_s", 0.0)
                agents[service]["agent_latency (s)"]["duration"] += e2e
                agents[service]["agent_latency (s)"]["runtime_duration"] += active_time
                agents[service]["cost ($)"]["billed_duration"] += active_time # Billed only for active compute
                orch_latency["agent_runtime_duration"] += active_time
                
        elif evt == "psutil_metrics_node":
            if service == "MLorchestrator":
                node_name = record.get("node_name")
                if node_name == "sync_s3_to_sandbox":
                    orch_latency["sync_s3"] += record.get("node_e2e_s", 0.0)
                    orch_cost["sync_s3_billed_duration"] += (record.get("node_e2e_s", 0.0) - record.get("wait_time_s", 0.0))
                    
        elif evt == "llm_call" and is_subagent:
            lat = record.get("wall_clock_s")
            if lat is None:
                lat = record.get("latency_ms", 0.0) / 1000.0
            agents[service]["agent_latency (s)"]["llm_latency"] += lat
            
            bd = agents[service]["cost ($)"]["llm_breakdown"]
            bd["input_tokens"] += record.get("input_tokens", 0)
            bd["cached_tokens"] += record.get("cached_tokens", 0)
            bd["output_tokens"] += record.get("output_tokens", 0)
            bd["reasoning_tokens"] += record.get("reasoning_tokens", 0)
            
        elif evt == "tool_call" and is_subagent:
            lat = record.get("latency_ms", 0.0) / 1000.0
            agents[service]["agent_latency (s)"]["total_tool_call_duration"] += lat

    # Post-process LLM Costs
    for service, data in agents.items():
        bd = data["cost ($)"]["llm_breakdown"]
        bd["input_tokens_non_cached"] = max(0, bd["input_tokens"] - bd["cached_tokens"])
        
        in_cost = (bd["input_tokens_non_cached"] / 1_000_000.0) * LLM_INPUT_COST_PER_1M
        cached_cost = 0.0 # Standard API doesn't charge for cache or has specific rate, assuming 0
        out_cost = (bd["output_tokens"] / 1_000_000.0) * LLM_OUTPUT_COST_PER_1M
        
        bd["input_cost"] = round(in_cost, 8)
        bd["cached_cost"] = round(cached_cost, 8)
        bd["output_cost"] = round(out_cost, 8)
        
        data["cost ($)"]["llm_total_cost"] = round(in_cost + cached_cost + out_cost, 8)
        
        # Round other fields
        data["agent_latency (s)"]["duration"] = round(data["agent_latency (s)"]["duration"], 4)
        data["agent_latency (s)"]["runtime_duration"] = round(data["agent_latency (s)"]["runtime_duration"], 4)
        data["agent_latency (s)"]["llm_latency"] = round(data["agent_latency (s)"]["llm_latency"], 4)
        data["agent_latency (s)"]["total_tool_call_duration"] = round(data["agent_latency (s)"]["total_tool_call_duration"], 4)
        data["cost ($)"]["billed_duration"] = round(data["cost ($)"]["billed_duration"], 8)
        data["cost ($)"]["peak_ram_gb"] = round(data["cost ($)"]["peak_ram_gb"], 4)

    # Round Orch Latency
    orch_latency["orch_runtime_duration"] = round(orch_latency["orch_runtime_duration"], 4)
    orch_latency["agent_runtime_duration"] = round(orch_latency["agent_runtime_duration"], 4)
    orch_latency["sync_s3"] = round(orch_latency["sync_s3"], 4)
    
    orch_cost["billed_duration"] = round(orch_cost["billed_duration"], 8)
    orch_cost["sync_s3_billed_duration"] = round(orch_cost["sync_s3_billed_duration"], 8)
    orch_cost["peak_ram_gb"] = round(orch_cost["peak_ram_gb"], 4)

    output = {
        "orch_latency (s)": orch_latency,
        "orch_cost ($)": orch_cost
    }
    
    for service, data in agents.items():
        clean_name = service.replace("_agent", "")
        output[clean_name] = data
        
    out_file = run_dir / "latency_and_cost.json"
    print(f"Metrics successfully written to {out_file}")
    
    # Also write to file
    out_file = run_dir / "latency_and_cost.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
