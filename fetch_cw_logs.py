import os
import sys
import yaml
import boto3
from pathlib import Path
from datetime import datetime, timedelta

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

def fetch_logs(log_group_name, service_name):
    print(f"\n==========================================================")
    print(f"   Logs for Service: {service_name}")
    print(f"   Log Group: {log_group_name}")
    print(f"==========================================================")
    
    logs_client = boto3.client("logs", region_name="ap-south-1")
    
    try:
        # Get log streams sorted by last event time
        streams_response = logs_client.describe_log_streams(
            logGroupName=log_group_name,
            orderBy="LastEventTime",
            descending=True,
            limit=3
        )
        
        streams = streams_response.get("logStreams", [])
        if not streams:
            print("No log streams found.")
            return

        # Fetch events from the most recent stream
        latest_stream = streams[0]["logStreamName"]
        print(f"Reading from stream: {latest_stream}")
        
        events_response = logs_client.get_log_events(
            logGroupName=log_group_name,
            logStreamName=latest_stream,
            limit=50,
            startFromHead=False
        )
        
        events = events_response.get("events", [])
        if not events:
            print("No log events found.")
            return
            
        for event in events:
            t = datetime.fromtimestamp(event['timestamp']/1000.0).strftime('%H:%M:%S')
            msg = event['message'].strip()
            print(f"[{t}] {msg}")
            
    except Exception as e:
        print(f"Failed to fetch logs: {e}")

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
    
    found_any = False
    for service in services:
        dir_path = root_dir / service
        info = get_agent_info(dir_path)
        if info:
            for name, arn in info:
                agent_hash = arn.split("/")[-1]
                log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_hash}-DEFAULT"
                fetch_logs(log_group_name, f"{service} ({name})")
                found_any = True
                
    if not found_any:
        print("No active Bedrock AgentCore agents found in config files.")

if __name__ == "__main__":
    main()
