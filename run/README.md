# Run directory
# This directory stores runtime log files produced by deployed agents.
# It is NOT committed to version control (see .gitignore).
#
# Files written here:
#   mcp_log.json  — JSONL execution log from the MCP Vector Store Server.
#                   Each line is a JSON record with: timestamp, tool, params,
#                   result_count, elapsed_ms, and (on error) error.
