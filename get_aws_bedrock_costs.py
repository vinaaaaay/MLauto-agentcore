#!/usr/bin/env python3
import boto3
import json
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Query AWS Cost Explorer for Amazon Bedrock / AgentCore Runtime charges.")
    parser.add_argument("--start-date", default="2026-07-01", help="Start date in YYYY-MM-DD format (default: 2026-07-01)")
    parser.add_argument("--end-date", default="2026-07-05", help="End date in YYYY-MM-DD format (default: 2026-07-05)")
    args = parser.parse_args()

    # Cost Explorer API must be queried from the global endpoint (us-east-1)
    ce_client = boto3.client('ce', region_name='us-east-1')

    print(f"Querying AWS Cost Explorer from {args.start_date} to {args.end_date} for Amazon Bedrock...")

    try:
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': args.start_date,
                'End': args.end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
            ]
        )
    except Exception as e:
        print(f"\n❌ Error querying Cost Explorer: {e}")
        print("\nNote: Make sure your AWS credentials are active and have 'ce:GetCostAndUsage' permissions.")
        return

    results = response.get("ResultsByTime", [])
    if not results:
        print("No cost records found for the given time period.")
        return

    print("\n" + "=" * 100)
    print(f"{'Date':<12} | {'Service':<40} | {'Usage Type':<20} | {'Quantity':<12} | {'Unblended Cost ($)':<18}")
    print("=" * 100)

    total_cost = 0.0
    for day in results:
        date_str = day.get("TimePeriod", {}).get("Start", "Unknown")
        groups = day.get("Groups", [])
        
        day_cost = 0.0
        for group in groups:
            keys = group.get("Keys", [])
            service = keys[0] if len(keys) > 0 else "N/A"
            usage_type = keys[1] if len(keys) > 1 else "N/A"
            
            metrics = group.get("Metrics", {})
            cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0.0))
            quantity = float(metrics.get("UsageQuantity", {}).get("Amount", 0.0))
            unit = metrics.get("UsageQuantity", {}).get("Unit", "")

            if cost > 0 or quantity > 0:
                print(f"{date_str:<12} | {service:<40} | {usage_type:<20} | {quantity:<12.4f} | ${cost:<17.6f}")
                day_cost += cost
                total_cost += cost
                
    print("=" * 100)
    print(f"{'GRAND TOTAL COST':<78} | ${total_cost:<17.6f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
