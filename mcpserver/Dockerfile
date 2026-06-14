FROM python:3.10-slim

# Set standard Python container variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_home

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies manifest and locks
COPY pyproject.toml uv.lock ./

# Copy mocks folder so the local wheels are available for installation
COPY mocks ./mocks

# Install dependencies using uv pip
RUN uv pip install --system --no-cache . --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download and cache the BGE embedding model weights inside the image
RUN python -c "from FlagEmbedding import FlagModel; FlagModel('BAAI/bge-base-en-v1.5')"

# Copy the rest of the application code
COPY . .

# Expose port 8080 (the default AgentCore runtime port)
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
