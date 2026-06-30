#!/usr/bin/env bash
# deploy.sh — Deploy MLorchestrator Agent to AWS Bedrock AgentCore.
#
# Prerequisites:
#   1. Run 'agentcore configure' inside MLorchestrator/ first (generates .bedrock_agentcore.yaml).
#      When prompted:
#        Entrypoint         -> app.py
#        Agent Name         -> mlorchestrator (or your preferred name)
#        Deployment Configs -> Select option 2 (Custom Container)
#        Dockerfile         -> Dockerfile
#        Execution Role     -> arn:aws:iam::235319806087:role/fame-agent-role
#        ECR Repo           -> Press [Enter] to auto-create
#   2. Fill in MLorchestrator/.env with AWS_REGION, S3_BUCKET_NAME, and target agent ARNs.
#
# Usage (from MLauto-agentcore/MLorchestrator/ directory):
#   chmod +x deploy.sh
#   ./deploy.sh --local-build
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    MLorchestrator Agent — Bedrock AgentCore Deploy"
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
    echo "  Entrypoint         -> app.py"
    echo "  Agent Name         -> mlorchestrator"
    echo "  Deployment Configs -> Select option 2 (Custom Container)"
    echo "  Dockerfile         -> Dockerfile"
    echo "  Execution Role     -> arn:aws:iam::235319806087:role/fame-agent-role"
    echo "  ECR Repo           -> Press [Enter] to auto-create"
    exit 1
fi

# --- 4. Deploy ---
echo ""
echo ">>> Deploying MLorchestrator..."
echo "    S3_BUCKET_NAME = ${S3_BUCKET_NAME}"
echo "    SEMANTIC_AGENT_ARN = ${SEMANTIC_AGENT_ARN:-}"
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
    --env S3_BUCKET_NAME="${S3_BUCKET_NAME}" \
    --env PERCEPTION_AGENT_ARN="${PERCEPTION_AGENT_ARN:-}" \
    --env SEMANTIC_AGENT_ARN="${SEMANTIC_AGENT_ARN:-}" \
    --env MEMORY_AGENT_ARN="${SEMANTIC_AGENT_ARN:-}" \
    --env CODING_AGENT_ARN="${CODING_AGENT_ARN:-}" \
    --env MCTS_HANDLER_ARN="${MCTS_HANDLER_ARN:-}" \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env LLM_MODEL="${LLM_MODEL:-}" \
    --env SANDBOX_URL="${SANDBOX_URL:-lambda:fame-sandbox-bastion}" \
    --env GATEWAY_LAMBDA_NAME="${GATEWAY_LAMBDA_NAME:-fame-sandbox-bastion}" \
    --env TARGET_IP="${TARGET_IP:-172.31.41.84}" \
    --env TARGET_PORT="${TARGET_PORT:-8080}"


echo ""
echo "=========================================================="
echo "    Successfully deployed MLorchestrator!"
echo "=========================================================="
