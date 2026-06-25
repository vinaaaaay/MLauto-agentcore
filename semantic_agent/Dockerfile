# Dockerfile for Semantic Agent


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

# Build package structure for semantic_agent to support package imports
COPY __init__.py agent.py prompts.py utils.py ./semantic_agent/
COPY common_local/ ./semantic_agent/common_local/

# Port 8080 — AgentCore default for agent runtimes
EXPOSE 8080

CMD ["python", "app.py"]
