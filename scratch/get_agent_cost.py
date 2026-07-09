import sys
import boto3

def main():
    try:
        client = boto3.client('ce')
        
        # We query for run_21 dates
        start_time = '2026-07-05T00:00:00Z'
        end_time = '2026-07-06T00:00:00Z'
        
        print(f"Querying AWS Cost Explorer from {start_time} to {end_time} (HOURLY granularity)...")
        print("Filtering for Amazon Bedrock AgentCore Memory and vCPU usage grouped by RESOURCE_ID...")
        
        response = client.get_cost_and_usage_with_resources(
            TimePeriod={
                'Start': start_time,
                'End': end_time
            },
            Granularity='HOURLY',
            Metrics=['UnblendedCost', 'UsageQuantity'],
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Bedrock AgentCore']
                }
            },
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'RESOURCE_ID'
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
        print("\nHourly Cost Breakdown by Agent (Resource ID):")
        print("=" * 145)
        print(f"{'Time Range (UTC)':<40} | {'Agent (Resource ID)':<50} | {'vCPU Cost ($)':<14} | {'Memory Cost ($)':<16} | {'Total Cost ($)':<15}")
        print("-" * 145)
        
        total_overall = 0.0
        
        for period in results:
            time_range = f"{period['TimePeriod']['Start']} to {period['TimePeriod']['End']}"
            groups = period.get('Groups', [])
            
            # Since a single agent might have separate vCPU and Memory entries, we group by resource_id
            agent_costs = {}
            
            for group in groups:
                if len(group['Keys']) < 2:
                    continue
                    
                resource_id = group['Keys'][0]
                usage_type = group['Keys'][1]
                
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if cost > 0:
                    agent_name = resource_id.split('/')[-1] if '/' in resource_id else resource_id
                    
                    if agent_name not in agent_costs:
                        agent_costs[agent_name] = {'cpu': 0.0, 'mem': 0.0}
                        
                    if 'vcpu' in usage_type.lower():
                        agent_costs[agent_name]['cpu'] += cost
                    elif 'memory' in usage_type.lower():
                        agent_costs[agent_name]['mem'] += cost

            for agent_name, costs in agent_costs.items():
                hour_total = costs['cpu'] + costs['mem']
                total_overall += hour_total
                print(f"{time_range:<40} | {agent_name:<50} | ${costs['cpu']:<13.6f} | ${costs['mem']:<15.6f} | ${hour_total:<14.6f}")
            
        print("=" * 145)
        print(f"{'TOTAL OVERALL':<111} | ${total_overall:<14.6f}")
        print("=" * 145)
        
    except Exception as e:
        print(f"\nError querying Cost Explorer: {e}", file=sys.stderr)
        print("Note: Grouping by RESOURCE_ID requires 'Hourly and Resource Level Data' to be enabled in your AWS Billing preferences.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
