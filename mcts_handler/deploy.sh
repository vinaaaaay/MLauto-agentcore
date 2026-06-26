#!/usr/bin/env bash
# deploy.sh — Deploy MCTS Handler Agent to AWS Bedrock AgentCore.
#
# Prerequisites:
#   1. Run 'agentcore configure' inside mcts_handler/ first (generates .bedrock_agentcore.yaml).
#      When prompted:
#        Entrypoint         -> app.py
#        Agent Name         -> mcts_handler (or your preferred name)
#        Deployment Configs -> Select option 2 (Custom Container)
#        Dockerfile         -> Dockerfile
#        Execution Role     -> arn:aws:iam::235319806087:role/fame-agent-role
#        ECR Repo           -> Press [Enter] to auto-create
#   2. Fill in mcts_handler/.env with AWS_REGION and S3_BUCKET_NAME.
#
# Usage (from MLauto-agentcore/mcts_handler/ directory):
#   chmod +x deploy.sh
#   ./deploy.sh --local-build
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "    MCTS Handler Agent — Bedrock AgentCore Deploy"
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
    echo "  Agent Name         -> mcts_handler"
    echo "  Deployment Configs -> Select option 2 (Custom Container)"
    echo "  Dockerfile         -> Dockerfile"
    echo "  Execution Role     -> arn:aws:iam::235319806087:role/fame-agent-role"
    echo "  ECR Repo           -> Press [Enter] to auto-create"
    exit 1
fi

# --- 4. Deploy ---
echo ""
echo ">>> Deploying MCTS Handler..."
echo "    S3_BUCKET_NAME = ${S3_BUCKET_NAME}"
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
    --env S3_BUCKET_NAME="${S3_BUCKET_NAME}"

echo ""
echo "=========================================================="
echo "    Successfully deployed MCTS Handler!"
echo "=========================================================="
