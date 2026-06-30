# Dockerfile for MLorchestrator Agent
FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install dependencies via uv
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

# Copy entrypoint and core orchestrator files
COPY app.py .
COPY config.yaml .
COPY graph.py .
COPY nodes.py .
COPY state.py .
COPY a2a_client.py .
COPY logging_config.py .
COPY sandbox_client.py .
COPY .env .

# Port 8080 — AgentCore default
EXPOSE 8080

CMD ["python", "app.py"]
