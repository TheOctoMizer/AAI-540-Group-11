#!/bin/bash
# ============================================================
#  nids_lambda_actions/deploy.sh
#  Creates (or updates) the nids-vm-shutdown Lambda function.
#
#  Usage:
#    chmod +x deploy.sh
#    ./deploy.sh
#
#  Prerequisites:
#    - AWS CLI configured with credentials that have permission to:
#        lambda:CreateFunction, lambda:UpdateFunctionCode,
#        lambda:GetFunction, iam:PassRole
#    - A SageMaker/Lambda execution role with ec2:StopInstances
#      and ec2:DescribeInstances permissions.
# ============================================================
set -euo pipefail

FUNCTION_NAME="nids-vm-shutdown"
REGION="${AWS_REGION:-us-east-1}"
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::539014262970:role/LabRole}"

# Tag values used to locate the Go server EC2 instance
INSTANCE_TAG_KEY="${INSTANCE_TAG_KEY:-Name}"
INSTANCE_TAG_VAL="${INSTANCE_TAG_VAL:-nids-go-server}"

# Set DRY_RUN=true to test permission/logic without stopping the VM
DRY_RUN_ENV="${DRY_RUN:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_FILE="$SCRIPT_DIR/nids_vm_shutdown.zip"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Package ---------------------------------------------------------------
log "Packaging Lambda function..."
cd "$SCRIPT_DIR"
zip -q -j "$ZIP_FILE" lambda_function.py
log "  Created: $ZIP_FILE ($(du -sh "$ZIP_FILE" | cut -f1))"

# ---- Deploy or Update ------------------------------------------------------
EXISTS=$(aws lambda get-function \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" \
  --query 'Configuration.FunctionName' \
  --output text 2>/dev/null || echo "NONE")

if [ "$EXISTS" = "$FUNCTION_NAME" ]; then
  log "Updating existing Lambda function '$FUNCTION_NAME'..."
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_FILE" \
    --region "$REGION" > /dev/null

  log "Updating environment variables..."
  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --environment "Variables={INSTANCE_TAG_KEY=$INSTANCE_TAG_KEY,INSTANCE_TAG_VAL=$INSTANCE_TAG_VAL,DRY_RUN=$DRY_RUN_ENV}" \
    --region "$REGION" > /dev/null

else
  log "Creating Lambda function '$FUNCTION_NAME'..."
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime python3.12 \
    --role "$ROLE_ARN" \
    --handler lambda_function.lambda_handler \
    --zip-file "fileb://$ZIP_FILE" \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={INSTANCE_TAG_KEY=$INSTANCE_TAG_KEY,INSTANCE_TAG_VAL=$INSTANCE_TAG_VAL,DRY_RUN=$DRY_RUN_ENV}" \
    --description "Stops nids-go-server EC2 on malicious traffic detection" \
    --region "$REGION" > /dev/null
fi

log "Waiting for Lambda to become Active..."
aws lambda wait function-active \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION"

log ""
log "============================================================"
log "  LAMBDA DEPLOYED"
log "============================================================"
log "  Function : $FUNCTION_NAME"
log "  Region   : $REGION"
log "  Role     : $ROLE_ARN"
log "  Tag key  : $INSTANCE_TAG_KEY"
log "  Tag val  : $INSTANCE_TAG_VAL"
log "  Dry-run  : $DRY_RUN_ENV"
log "============================================================"
log ""
log "  Test it:"
log "    aws lambda invoke \\"
log "      --function-name $FUNCTION_NAME \\"
log "      --payload '{\"attack_type\":\"DDoS\",\"confidence\":0.99,\"mse_error\":0.08,\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}' \\"
log "      --cli-binary-format raw-in-base64-out \\"
log "      --region $REGION \\"
log "      response.json && cat response.json"
