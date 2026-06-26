# Dockerfile for MCTS Handler Agent
FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install dependencies via uv
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

# Copy source code files
COPY app.py .
COPY node_state.py .
COPY tree_store.py .
COPY .env .

# Port 8080 — AgentCore default
EXPOSE 8080

CMD ["python", "app.py"]
