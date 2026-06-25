# Dockerfile for MCP Vector Store Server
# Follows the same pattern as FAME-fork/bedrockAC/workflows/react_distributed agents.
#
# Build context: MLauto-agentcore/ (project root — one level above mcpserver/)
# Example build command:
#   docker build -f mcpserver/Dockerfile -t mcp-vector-store .

FROM python:3.12-slim

# Install uv from the official Astral image (same as react_distributed agents)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install all Python dependencies via uv (fast, no-cache build)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy MCP server application code
COPY mcp_server.py .
COPY app.py .

# Port 8080 — AgentCore default for agent runtimes
EXPOSE 8080

CMD ["python", "app.py"]
