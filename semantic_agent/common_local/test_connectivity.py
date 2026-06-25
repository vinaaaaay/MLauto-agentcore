import os
import sys
import boto3

# Add parent directory to path to import mcp module correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcp.mcp_client import MCPClient
except ImportError:
    # Fallback for when running from within the directory
    from custom_mcp.mcp_client import MCPClient

def get_lambda_url(function_name: str, region: str = "ap-south-1") -> str:
    """Retrieve Lambda Function URL."""
    try:
        client = boto3.client("lambda", region_name=region)
        response = client.get_function_url_config(FunctionName=function_name)
        return response["FunctionUrl"]
    except Exception as e:
        print(f"❌ Failed to get URL for {function_name}: {str(e)}")
        return "PLACEHOLDER_URL"

def test_server(name: str, url: str, test_tools: dict = None):
    print(f"\n{'='*20} Testing {name} {'='*20}")
    print(f"URL: {url}")
    
    if "PLACEHOLDER" in url:
        print("⚠️  Skipping connection test: URL is a placeholder")
        return

    client = MCPClient(url, client_name="test-connectivity-client")
    try:
        # 1. List Tools
        print("\n1. Listing Tools...")
        tools = client.list_tools()
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', '')[:50]}...")

        # 2. Invoke Tools
        if test_tools:
            print("\n2. Invoking Test Tools...")
            for tool_name, args in test_tools.items():
                print(f"\n  > Invoking {tool_name} with {args}...")
                try:
                    result = client.call_tool(tool_name, args)
                    print("  ✅ Success!")
                    print("  Result Preview:", str(result)[:200] + "..." if len(str(result)) > 200 else result)
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")

    except Exception as e:
        print(f"❌ Connection Failed: {str(e)}")
    finally:
        client.close()

def main():
    # Arxiv Consolidated Server - known URL
    arxiv_url = "https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/"
    
    # Log Consolidated Server - fetch URL from ARN via boto3
    print("Fetching Log Consolidated Lambda Function URL...")
    log_url = get_lambda_url("log_consolidated_lambda")
    
    # Define test cases for each server
    arxiv_tests = {
        "download_article": {"title": "Attention Is All You Need"},
    }
    
    log_tests = {
        "create_bar_chart": {
            "title": "Test Chart",
            "x_label": "Category",
            "y_label": "Value",
            "categories": ["A", "B", "C"],
            "values": [10, 20, 30]
        },
    }

    test_server("Arxiv Consolidated Server", arxiv_url, arxiv_tests)
    test_server("Log Consolidated Server", log_url, log_tests)


if __name__ == "__main__":
    main()
