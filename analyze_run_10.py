import os
import sys
import json
import yaml
import boto3
from pathlib import Path
from datetime import datetime, timezone

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

def fetch_metrics(log_group_name, service_name, start_time_epoch, end_time_epoch):
    logs_client = boto3.client("logs", region_name="ap-south-1")
    metrics_records = []
    
    try:
        # Filter log events using filter_log_events to search across all streams in the time window
        paginator = logs_client.get_paginator('filter_log_events')
        pages = paginator.paginate(
            logGroupName=log_group_name,
            startTime=int(start_time_epoch * 1000),
            endTime=int(end_time_epoch * 1000),
            filterPattern='psutil_metrics'
        )
        
        for page in pages:
            events = page.get("events", [])
            for event in events:
                msg = event['message'].strip()
                # Extract the JSON payload
                try:
                    # Find start of JSON
                    idx = msg.find('{')
                    if idx != -1:
                        data = json.loads(msg[idx:])
                        data["_service"] = service_name
                        data["_timestamp"] = event['timestamp']
                        metrics_records.append(data)
                except Exception:
                    pass
    except Exception as e:
        print(f"Failed to fetch logs for {service_name}: {e}")
        
    return metrics_records

def main():
    root_dir = Path(__file__).resolve().parent
    services = [
        "MLorchestrator",
        "coder_agent",
        "perception_agent",
        "semantic_agent",
        "mcts_handler",
        "mcpserver"
    ]
    
    # We define the UTC time window for run_10:
    # Started around 10:39:00 UTC, ended around 11:14:00 UTC on 2026-07-02
    start_time = datetime(2026, 7, 2, 10, 38, 0, tzinfo=timezone.utc).timestamp()
    end_time = datetime(2026, 7, 2, 11, 16, 0, tzinfo=timezone.utc).timestamp()
    
    all_metrics = []
    for service in services:
        dir_path = root_dir / service
        info = get_agent_info(dir_path)
        if info:
            for name, arn in info:
                agent_hash = arn.split("/")[-1]
                log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_hash}-DEFAULT"
                records = fetch_metrics(log_group_name, f"{service} ({name})", start_time, end_time)
                all_metrics.extend(records)
                
    # Also parse from the local logs.txt of run_10 for mlorchestrator just in case some logs are missing in CW
    local_logs_path = root_dir / "runs" / "run_10" / "logs.txt"
    if local_logs_path.exists():
        with open(local_logs_path, "r", encoding="utf-8") as f:
            for line in f:
                if "psutil_metrics" in line:
                    try:
                        idx = line.find('{')
                        if idx != -1:
                            data = json.loads(line[idx:])
                            # Deduplicate by span_id/timestamp
                            if not any(x.get("span_id") == data.get("span_id") and x.get("event_type") == data.get("event_type") and x.get("node_name") == data.get("node_name") for x in all_metrics):
                                data["_service"] = "MLorchestrator (local)"
                                all_metrics.append(data)
                    except Exception:
                        pass

    # Deduplicate raw metrics records across streams (runtime-logs and otel-rt-logs)
    unique_metrics = []
    seen_keys = set()
    for record in all_metrics:
        event_type = record.get("event_type")
        timestamp = record.get("timestamp") or record.get("_timestamp")
        name = record.get("node_name") or record.get("graph_name") or "unknown"
        e2e = record.get("graph_e2e_s") or record.get("node_e2e_s") or 0.0
        
        key = (event_type, name, timestamp, round(e2e, 4))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_metrics.append(record)
            
    all_metrics = unique_metrics

    # Process and summarize
    print(f"\n==========================================================")
    print(f"      AGENTCORE RUNTIME METRICS ANALYSIS - RUN 10 (DEDUPLICATED)")
    print(f"==========================================================\n")
    
    summary = {}
    node_breakdown = []
    
    for record in all_metrics:
        event_type = record.get("event_type")
        service = record.get("_service")
        
        if event_type == "psutil_metrics_graph":
            if service not in summary:
                summary[service] = {
                    "e2e_s": 0.0,
                    "cpu_s": 0.0,
                    "wait_s": 0.0,
                    "peak_ram_gb": 0.0,
                    "io_read_mb": 0.0,
                    "io_write_mb": 0.0,
                    "invocations": 0
                }
            summary[service]["e2e_s"] += record.get("graph_e2e_s", 0.0)
            summary[service]["cpu_s"] += record.get("active_cpu_s", 0.0)
            summary[service]["wait_s"] += record.get("wait_time_s", 0.0)
            summary[service]["peak_ram_gb"] = max(summary[service]["peak_ram_gb"], record.get("peak_RAM_GB", 0.0))
            summary[service]["io_read_mb"] += record.get("io_read_MB", 0.0)
            summary[service]["io_write_mb"] += record.get("io_write_MB", 0.0)
            summary[service]["invocations"] += 1
            
        elif event_type == "psutil_metrics_node":
            node_breakdown.append({
                "service": service,
                "node_name": record.get("node_name"),
                "e2e_s": record.get("node_e2e_s", 0.0),
                "cpu_s": record.get("active_cpu_s", 0.0),
                "wait_s": record.get("wait_time_s", 0.0),
                "peak_ram_gb": record.get("peak_RAM_GB", 0.0),
                "io_read_mb": record.get("io_read_MB", 0.0),
                "io_write_mb": record.get("io_write_MB", 0.0)
            })

    print("SERVICE SUMMARY:")
    print("-" * 115)
    print(f"{'Service':<30} | {'Invocations':<12} | {'E2E Time (s)':<14} | {'CPU Time (s)':<14} | {'Wait Time (s)':<14} | {'Peak RAM (GB)':<14} | {'I/O W (MB)':<10}")
    print("-" * 115)
    for service, metrics in summary.items():
        print(f"{service:<30} | {metrics['invocations']:<12} | {metrics['e2e_s']:<14.4f} | {metrics['cpu_s']:<14.4f} | {metrics['wait_s']:<14.4f} | {metrics['peak_ram_gb']:<14.4f} | {metrics['io_write_mb']:<10.4f}")
    print("-" * 115)
    
    print("\nORCHESTRATOR NODE BREAKDOWN:")
    print("-" * 110)
    print(f"{'Node Name':<30} | {'Count':<6} | {'Total E2E (s)':<14} | {'Total CPU (s)':<14} | {'Total Wait (s)':<14} | {'Max RAM (GB)':<12}")
    print("-" * 110)
    
    # Aggregate node breakdown by node_name
    node_summary = {}
    for node in node_breakdown:
        name = node["node_name"]
        if name not in node_summary:
            node_summary[name] = {
                "count": 0,
                "e2e_s": 0.0,
                "cpu_s": 0.0,
                "wait_s": 0.0,
                "max_ram_gb": 0.0
            }
        node_summary[name]["count"] += 1
        node_summary[name]["e2e_s"] += node["e2e_s"]
        node_summary[name]["cpu_s"] += node["cpu_s"]
        node_summary[name]["wait_s"] += node["wait_s"]
        node_summary[name]["max_ram_gb"] = max(node_summary[name]["max_ram_gb"], node["peak_ram_gb"])
        
    for name, s in sorted(node_summary.items(), key=lambda x: x[1]["e2e_s"], reverse=True):
        print(f"{name:<30} | {s['count']:<6} | {s['e2e_s']:<14.4f} | {s['cpu_s']:<14.4f} | {s['wait_s']:<14.4f} | {s['max_ram_gb']:<12.4f}")
    print("-" * 110)

if __name__ == "__main__":
    main()
