#!/usr/bin/env python3
import json
import os
import glob
import argparse

CPU_HOURLY_RATE = 0.0895
RAM_HOURLY_RATE = 0.00945

def calculate_costs(duration_s, ram_gb, vcpu_count):
    duration_h = duration_s / 3600.0
    cpu_cost = duration_h * vcpu_count * CPU_HOURLY_RATE
    ram_cost = duration_h * ram_gb * RAM_HOURLY_RATE
    return cpu_cost, ram_cost

def process_run(json_path, vcpu_count, ram_gb):
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = {}
    total_cpu_cost = 0.0
    total_ram_cost = 0.0
    
    # Process Orchestrator
    if "orch_latency (s)" in data:
        dur = data["orch_latency (s)"].get("orch_e2e_duration", 0)
        ram = ram_gb
        cpu_cost, ram_cost = calculate_costs(dur, ram, vcpu_count)
        results["orchestrator"] = {"cpu": cpu_cost, "ram": ram_cost, "total": cpu_cost + ram_cost}
        total_cpu_cost += cpu_cost
        total_ram_cost += ram_cost

    # Process Agents
    agents = ["perception", "semantic", "mcts", "coder"]
    for agent in agents:
        if agent in data:
            dur = data[agent].get("agent_latency (s)", {}).get("total_e2e_duration", 0)
            ram = ram_gb
            cpu_cost, ram_cost = calculate_costs(dur, ram, vcpu_count)
            results[agent] = {"cpu": cpu_cost, "ram": ram_cost, "total": cpu_cost + ram_cost}
            total_cpu_cost += cpu_cost
            total_ram_cost += ram_cost

    # Process MCP Server
    if "mcpserver" in data and "semantic" in data:
        # MCP server duration is semantic agent's tool call duration
        mcp_dur = data["semantic"].get("agent_latency (s)", {}).get("total_tool_call_duration", 0)
        mcp_ram = ram_gb
        cpu_cost, ram_cost = calculate_costs(mcp_dur, mcp_ram, vcpu_count)
        results["mcpserver"] = {"cpu": cpu_cost, "ram": ram_cost, "total": cpu_cost + ram_cost}
        total_cpu_cost += cpu_cost
        total_ram_cost += ram_cost

    return results, total_cpu_cost, total_ram_cost

def main():
    parser = argparse.ArgumentParser(description="Calculate CPU and RAM compute costs for runs.")
    parser.add_argument("run", nargs="?", help="Specific run name (e.g. run_21) or path to latency_and_cost.json. If omitted, runs all.")
    parser.add_argument("--runs-dir", default="runs", help="Path to the runs directory")
    parser.add_argument("--vcpus", type=float, default=2.0, help="Number of vCPUs per component")
    parser.add_argument("--ram", type=float, default=8.0, help="Fixed RAM per component in GB")
    args = parser.parse_args()

    if args.run:
        if os.path.isfile(args.run):
            files = [args.run]
        else:
            potential_file = os.path.join(args.runs_dir, args.run, "latency_and_cost.json")
            if os.path.isfile(potential_file):
                files = [potential_file]
            else:
                potential_file2 = os.path.join(args.run, "latency_and_cost.json")
                if os.path.isfile(potential_file2):
                    files = [potential_file2]
                else:
                    print(f"Error: Could not find latency_and_cost.json for '{args.run}'")
                    return
    else:
        search_pattern = os.path.join(args.runs_dir, "*", "latency_and_cost.json")
        files = sorted(glob.glob(search_pattern))

    if not files:
        print("No latency_and_cost.json files found.")
        return

    for file_path in files:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
        results, total_cpu, total_ram = process_run(file_path, args.vcpus, args.ram)
        
        print(f"\nRun: {run_name}")
        print("-" * 75)
        print(f"{'Component':<15} | {'CPU Cost ($)':<16} | {'RAM Cost ($)':<16} | {'Total Cost ($)':<14}")
        print("-" * 75)
        
        for comp, costs in results.items():
            print(f"{comp:<15} | ${costs['cpu']:<15.6f} | ${costs['ram']:<15.6f} | ${costs['total']:<14.6f}")
        
        print("-" * 75)
        grand_total = total_cpu + total_ram
        print(f"{'TOTALS':<15} | ${total_cpu:<15.6f} | ${total_ram:<15.6f} | ${grand_total:<14.6f}\n")

if __name__ == "__main__":
    main()
