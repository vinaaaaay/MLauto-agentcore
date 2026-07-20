import re
from pathlib import Path

def analyze_a2a_calls(run_dir_str):
    cw_logs = Path(run_dir_str) / "cw_logs.txt"
    if not cw_logs.exists():
        print(f"File not found: {cw_logs}")
        return
        
    # Matches: [A2A Call #X] AgentName (skill_name) completed in 0.00s
    pattern = re.compile(r"\[A2A Call #\d+\] ([\w]+) \(([^)]+)\) completed in ([\d\.]+)s")
    
    total_time = 0.0
    breakdown = {}
    count_breakdown = {}
    
    with open(cw_logs, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                agent = m.group(1)
                skill = m.group(2)
                dur = float(m.group(3))
                
                key = f"{agent} ({skill})"
                breakdown[key] = breakdown.get(key, 0.0) + dur
                count_breakdown[key] = count_breakdown.get(key, 0) + 1
                total_time += dur
                
    print(f"--- A2A HTTP Call Sum for {run_dir_str} ---")
    print(f"Total A2A API blocking time: {total_time:.4f}s")
    print("\nBreakdown by Agent (Skill):")
    for k, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        count = count_breakdown[k]
        print(f"  {k:<35}: {v:>8.4f}s  ({count} calls)")
    print("="*50)

if __name__ == "__main__":
    analyze_a2a_calls("runs/run_22")
    analyze_a2a_calls("runs/run_28")
