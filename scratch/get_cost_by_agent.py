import sys
import boto3

def main():
    try:
        client = boto3.client('ce')
        
        # get_cost_and_usage_with_resources only supports DAILY or MONTHLY granularity
        start_time = '2026-07-05'
        end_time = '2026-07-07'
        
        print(f"Querying AWS Cost Explorer from {start_time} to {end_time} (DAILY) by RESOURCE_ID...")
        
        response = client.get_cost_and_usage_with_resources(
            TimePeriod={
                'Start': start_time,
                'End': end_time
            },
            Granularity='DAILY',
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Bedrock AgentCore']
                }
            },
            Metrics=['UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'USAGE_TYPE'
                },
                {
                    'Type': 'DIMENSION',
                    'Key': 'RESOURCE_ID'
                }
            ]
        )
        
        results = response.get('ResultsByTime', [])
        if not results:
            print("No cost records returned.")
            return

        print("\nAgent Cost & Usage Breakdown by Resource ID (vCPU & Memory only):")
        print("=" * 170)
        print(f"{'Date':<15} | {'Resource ID (Agent)':<65} | {'Usage Type':<45} | {'Cost ($)':<12} | {'Usage Qty':<15}")
        print("-" * 170)
        
        total_overall = 0.0
        
        for period in results:
            date = period['TimePeriod']['Start']
            groups = period.get('Groups', [])
            
            for group in groups:
                usage_type = group['Keys'][0]
                resource_id = group['Keys'][1]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                usage_qty = float(group['Metrics']['UsageQuantity']['Amount'])
                
                is_cpu_or_mem = ('vcpu' in usage_type.lower() or 'memory' in usage_type.lower())
                
                if is_cpu_or_mem and cost > 0.0:
                    # simplify resource_id if it's an ARN
                    if "agent/" in resource_id:
                        resource_id = resource_id.split("agent/")[-1]
                    
                    print(f"{date:<15} | {resource_id:<65} | {usage_type:<45} | ${cost:<11.6f} | {usage_qty:<15.6f}")
                    total_overall += cost
                    
        print("=" * 170)
        print(f"Total Compute Cost (vCPU & Memory) for all agents: ${total_overall:.6f}")
        
    except Exception as e:
        print(f"Error querying Cost Explorer: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
