import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Sum up the exact polling times for the Coder agent in a run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run output directory (e.g. runs/run_22)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: run directory '{args.run_dir}' does not exist.")
        return

    cw_logs_path = run_dir / "cw_logs.txt"
    if not cw_logs_path.exists():
        print(f"Error: {cw_logs_path} not found.")
        return

    # Regex to extract the duration from the A2A call log line
    # Example line: ... [A2A Call #2] CodingAgent (check_status) completed in 0.11s
    check_status_pattern = re.compile(r"CodingAgent \(check_status\) completed in ([\d\.]+)s")
    generate_and_run_pattern = re.compile(r"CodingAgent \(generate_and_run\) completed in ([\d\.]+)s")

    total_check_status_time = 0.0
    check_status_count = 0
    
    total_generate_time = 0.0
    generate_count = 0

    with open(cw_logs_path, "r", encoding="utf-8") as f:
        for line in f:
            # Check for check_status
            match_check = check_status_pattern.search(line)
            if match_check:
                duration = float(match_check.group(1))
                total_check_status_time += duration
                check_status_count += 1
                continue
                
            # Check for generate_and_run
            match_gen = generate_and_run_pattern.search(line)
            if match_gen:
                duration = float(match_gen.group(1))
                total_generate_time += duration
                generate_count += 1

    print(f"=== Polling Analysis for {run_dir.name} ===")
    print(f"check_status calls:       {check_status_count}")
    print(f"check_status total time:  {total_check_status_time:.4f}s")
    print(f"Average time per poll:    {total_check_status_time / check_status_count if check_status_count else 0:.4f}s")
    print("-" * 40)
    print(f"generate_and_run calls:   {generate_count}")
    print(f"generate_and_run time:    {total_generate_time:.4f}s")
    print("=========================================")

if __name__ == "__main__":
    main()
