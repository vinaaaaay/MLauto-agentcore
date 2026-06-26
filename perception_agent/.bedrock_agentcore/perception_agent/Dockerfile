# Dockerfile for Perception Agent

FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install all Python dependencies via uv
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

# Copy entrypoint and local packages
COPY app.py .
COPY common_local/ ./common_local/

# Build package structure for perception_agent to support package imports
COPY __init__.py agent.py state.py prompts.py utils.py sandbox_client.py ./perception_agent/
COPY common_local/ ./perception_agent/common_local/
COPY tools_registry/ ./perception_agent/tools_registry/

# Port 8080 — AgentCore default for agent runtimes
EXPOSE 8080

CMD ["python", "app.py"]
