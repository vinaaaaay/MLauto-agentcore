#!/usr/bin/env bash
# deploy.sh — Deploy Perception Agent to AWS Bedrock AgentCore.
#
# Prerequisites:
#   1. Run 'agentcore configure' inside perception_agent/ first (generates .bedrock_agentcore.yaml).
#      When prompted:
#        Entrypoint         -> app.py
#        Agent Name         -> perception_agent (or your preferred name)
#        Deployment Configs -> Select option 2 (Custom Container)
#        Dockerfile         -> Dockerfile
#        Execution Role     -> arn:aws:iam::235319806087:role/AmazonBedrockAgentCoreSDKRuntime-ap-south-1-76abb37315
#        ECR Repo           -> Press [Enter] to auto-create
#   2. Fill in perception_agent/.env with all required variables.
#
# Usage (from MLauto-agentcore/perception_agent/ directory):
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Optional flags:
#   --local-build   Build Docker image locally instead of via AWS CodeBuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    Perception Agent — Bedrock AgentCore Deploy"
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
    echo "         Create .env from the template and fill in your values."
fi

# --- 2. Validate required environment variables ---
REQUIRED_VARS=("AWS_REGION" "SANDBOX_URL")
MISSING_VARS=0

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable '$var' is not set."
        MISSING_VARS=$((MISSING_VARS + 1))
    fi
done

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set."
    MISSING_VARS=$((MISSING_VARS + 1))
fi

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
    echo "  Entrypoint         -> app.py"
    echo "  Agent Name         -> perception_agent"
    echo "  Deployment Configs -> Select option 2 (Custom Container)"
    echo "  Dockerfile         -> Dockerfile"
    echo "  Execution Role     -> arn:aws:iam::235319806087:role/AmazonBedrockAgentCoreSDKRuntime-ap-south-1-76abb37315"
    echo "  ECR Repo           -> Press [Enter] to auto-create"
    exit 1
fi

# --- 4. Deploy ---
echo ""
echo ">>> Deploying Perception Agent..."
echo "    SANDBOX_URL = ${SANDBOX_URL}"
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
    --env AWS_REGION="${AWS_REGION}" \
    --env AWS_DEFAULT_REGION="${AWS_REGION}" \
    --env SANDBOX_URL="${SANDBOX_URL}" \
    --env GATEWAY_LAMBDA_NAME="${GATEWAY_LAMBDA_NAME:-fame-sandbox-bastion}" \
    --env TARGET_IP="${TARGET_IP:-172.31.41.59}" \
    --env TARGET_PORT="${TARGET_PORT:-8080}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env LLM_MODEL="${LLM_MODEL:-}"

echo ""
echo "=========================================================="
echo "    Successfully deployed Perception Agent!"
echo ""
echo "  Next steps:"
echo "  1. Copy the agent ARN from the output above."
echo "  2. Configure your orchestrator to use this ARN."
echo "=========================================================="
