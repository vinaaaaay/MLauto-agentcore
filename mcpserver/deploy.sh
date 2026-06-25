#!/usr/bin/env bash
# deploy.sh — Deploy MCP Vector Store Server to AWS Bedrock AgentCore.
#
# Prerequisites:
#   1. Run 'agentcore configure' inside mcpserver/ first (generates .bedrock_agentcore.yaml).
#   2. Fill in mcpserver/.env with your S3_BUCKET_NAME, S3_KEY, and AWS_REGION.
#   3. Ensure the agentcore CLI is available (activate your venv or install globally).
#
# Usage (from MLauto-agentcore/mcpserver/ directory):
#   chmod +x deploy.sh
#   ./deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    MCP Vector Store Server — Bedrock AgentCore Deploy"
echo "=========================================================="

# --- 1. Load environment variables from .env ---
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo ">>> Loading environment variables from .env..."
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "WARNING: .env file not found at $ENV_FILE"
    echo "         Copy .env.example to .env and fill in your values."
fi

# --- 2. Validate required environment variables ---
REQUIRED_VARS=("AWS_REGION" "S3_BUCKET_NAME")
MISSING_VARS=0

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable '$var' is not set."
        MISSING_VARS=$((MISSING_VARS + 1))
    fi
done

if [ $MISSING_VARS -gt 0 ]; then
    echo ""
    echo "Please set the missing variables in $ENV_FILE and re-run."
    exit 1
fi

# --- 3. Check for .bedrock_agentcore.yaml ---
if [ ! -f "$SCRIPT_DIR/.bedrock_agentcore.yaml" ]; then
    echo ""
    echo "ERROR: .bedrock_agentcore.yaml not found."
    echo ""
    echo "Please run 'agentcore configure' first:"
    echo "  cd $SCRIPT_DIR && agentcore configure"
    echo ""
    echo "When prompted:"
    echo "  Entrypoint         -> Press [Enter]  (uses default app.py)"
    echo "  Agent Name         -> Enter a name   (e.g. mcp-vector-store)"
    echo "  Dependency File    -> Press [Enter]  (uses default requirements.txt)"
    echo "  Deployment Configs -> Select option 2 (Custom Container)"
    echo "  Execution Role     -> Press [Enter]  (auto create) or provide ARN"
    echo "  ECR Repo           -> Press [Enter]  (auto create) or provide ARN"
    exit 1
fi

# --- 4. Deploy ---
echo ""
echo ">>> Deploying MCP Vector Store Server..."
echo ""

DEPLOY_FLAGS=""
if [[ "${1:-}" == "--local-build" ]]; then
    echo "Using local build..."
    DEPLOY_FLAGS="--local-build"
    shift
else
    echo "Using remote AWS CodeBuild (cross-platform linux/arm64)..."
fi

agentcore deploy $DEPLOY_FLAGS "$@" \
    --env S3_BUCKET_NAME="${S3_BUCKET_NAME}" \
    --env S3_KEY="${S3_KEY:-tools_registry.zip}" \
    --env AWS_REGION="${AWS_REGION}"

echo ""
echo "=========================================================="
echo "    Successfully deployed MCP Vector Store Server!"
echo "=========================================================="
