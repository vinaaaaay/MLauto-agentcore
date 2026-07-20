import sys
import boto3

def main():
    try:
        client = boto3.client('ce')
        
        # We query from 06:00 to 09:00 UTC on July 9th.
        # This will return three 1-hour periods:
        # - 06:00 to 07:00 (which covers the start of Run 22 at 06:49)
        # - 07:00 to 08:00
        # - 08:00 to 09:00 (which covers the end of Run 22 at 08:50)
        start_time = '2026-07-09T06:00:00Z'
        end_time = '2026-07-09T09:00:00Z'
        
        print(f"Querying AWS Cost Explorer from {start_time} to {end_time} (HOURLY granularity)...")
        print("Filtering for Amazon Bedrock AgentCore Memory and vCPU usage only...")
        
        response = client.get_cost_and_usage(
            TimePeriod={
                'Start': start_time,
                'End': end_time
            },
            Granularity='HOURLY',
            Metrics=['UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                },
                {
                    'Type': 'DIMENSION',
                    'Key': 'USAGE_TYPE'
                }
            ]
        )
        
        results = response.get('ResultsByTime', [])
        if not results:
            print("No hourly cost records returned.")
            return

        # Print header
        print("\nHourly Cost & Usage Breakdown (vCPU & Memory combined):")
        print("=" * 155)
        print(f"{'Time Range (UTC)':<40} | {'vCPU Cost ($)':<14} | {'vCPU Qty (hrs)':<16} | {'Memory Cost ($)':<16} | {'Memory Qty (GB-hrs)':<20} | {'Total Cost ($)':<15}")
        print("-" * 155)
        
        total_overall = 0.0
        
        for period in results:
            time_range = f"{period['TimePeriod']['Start']} to {period['TimePeriod']['End']}"
            groups = period.get('Groups', [])
            
            mem_cost = 0.0
            mem_qty = 0.0
            cpu_cost = 0.0
            cpu_qty = 0.0
            
            for group in groups:
                service = group['Keys'][0]
                usage_type = group['Keys'][1]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                usage_qty = float(group['Metrics']['UsageQuantity']['Amount'])
                
                is_agentcore = (service == 'Amazon Bedrock AgentCore')
                if is_agentcore:
                    if 'vcpu' in usage_type.lower():
                        cpu_cost += cost
                        cpu_qty += usage_qty
                    elif 'memory' in usage_type.lower():
                        mem_cost += cost
                        mem_qty += usage_qty
            
            hour_total = cpu_cost + mem_cost
            total_overall += hour_total
            
            print(f"{time_range:<40} | ${cpu_cost:<13.6f} | {cpu_qty:<16.6f} | ${mem_cost:<15.6f} | {mem_qty:<20.6f} | ${hour_total:<14.6f}")
            
        print("=" * 155)
        print(f"{'TOTAL OVERALL (vCPU & Memory Combined)':<119} | ${total_overall:<14.6f}")
        print("=" * 155)
        
    except Exception as e:
        print(f"Error querying Cost Explorer: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
