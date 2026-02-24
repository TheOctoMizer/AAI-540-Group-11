#!/bin/bash
# ============================================================
#  deploy_all.sh - Full NIDS Stack Deployment
#  Steps:
#    0. Deploy Lambda      - nids-vm-shutdown (shutdown action)
#    1. Deploy Go server     - t4g.micro EC2 (subnet A, ARM64)
#    2. Deploy XGBoost       - SageMaker     (ml.m5.large)
#    3. Deploy Rust NIDS     - t2.small EC2  (subnet B, x86_64)
#
#  All EC2 instances use different subnets (multi-AZ).
#  Rust NIDS communicates with Go server via PRIVATE IP.
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO_DIR="$ROOT_DIR/go_server"
NIDS_DIR="$ROOT_DIR/ai_nids_rust"
DEPLOY_DIR="$ROOT_DIR/nids_sagemaker_deploy"
MODELS_DIR="$NIDS_DIR/models"

REGION="us-east-1"
AMI_X86="ami-0f3caa1cf4417e51b"    # Amazon Linux 2023 x86_64
AMI_ARM64="ami-0bea3ccc607167c10"  # Amazon Linux 2023 arm64 (for t4g)
KEY_NAME="vockey"
KEY_PATH="${HOME}/.ssh/labsuser.pem"
ROLE_ARN="arn:aws:iam::539014262970:role/LabRole"
BUCKET="nids-mlops-models"
XGBOOST_ENDPOINT="nids-xgboost"
LAMBDA_FUNCTION="nids-vm-shutdown"
LAMBDA_DIR="$ROOT_DIR/nids_lambda_actions"

# ------ Helpers ---------------------------------------------------------------------------------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_ssh() {
  local ip=$1
  log "Waiting for SSH at $ip..."
  for i in $(seq 1 40); do
    if ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
         ec2-user@"$ip" 'exit' 2>/dev/null; then
      log "  SSH ready (attempt $i)"; return 0
    fi
    sleep 6
  done
  die "SSH timed out for $ip"
}

# ------ Discover all subnets ------------------------------------------------------------------------------------------------------------
SUBNETS=()
while IFS= read -r _subnet; do
  [[ -n "$_subnet" ]] && SUBNETS+=("$_subnet")
done < <(aws ec2 describe-subnets --region "$REGION" \
  --query 'Subnets[*].SubnetId' --output text | tr '\t' '\n')
log "Found ${#SUBNETS[@]} subnets: ${SUBNETS[*]}"
[ "${#SUBNETS[@]}" -ge 2 ] || die "Need at least 2 subnets"

SUBNET_GO="${SUBNETS[0]}"    # Go server subnet
SUBNET_NIDS="${SUBNETS[1]}"  # Rust NIDS subnet

# ------ Security Group ------------------------------------------------------------------------------------------------------------------------------
log "Checking for security group 'go-server-sg'..."
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")

SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=go-server-sg Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  log "  Creating new security group 'go-server-sg'..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "go-server-sg" \
    --description "NIDS stack - SSH, 8080, 3000" \
    --vpc-id "$VPC_ID" --query 'GroupId' --output text --region "$REGION")
  
  log "  Authorizing ingress: 22, 8080, 3000 from 0.0.0.0/0..."
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22   --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8080 --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 3000 --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
fi
log "Security group: $SG_ID"

# Ensure SSH key has correct permissions
if [ -f "$KEY_PATH" ]; then
  chmod 400 "$KEY_PATH"
fi

# ============================================================
#  STEP 0 --- Lambda Shutdown Action
# ============================================================
log ""
log "------------------------------------------------------------------------------------------------------------------------------------"
log "  STEP 0: Lambda Shutdown Action (nids-vm-shutdown)"
log "------------------------------------------------------------------------------------------------------------------------------------"

log "Deploying / updating Lambda function '$LAMBDA_FUNCTION'..."
LAMBDA_ROLE_ARN="$ROLE_ARN" \
AWS_REGION="$REGION" \
INSTANCE_TAG_KEY="Name" \
INSTANCE_TAG_VAL="nids-go-server" \
DRY_RUN="false" \
  bash "$LAMBDA_DIR/deploy.sh"
log "  --- Lambda '$LAMBDA_FUNCTION' deployed"

# ============================================================
#  STEP 1 --- Go Server (t4g.micro)
# ============================================================
log ""
log "------------------------------------------------------------------------------------------------------------------------------------"
log "  STEP 1: Go Server (t4g.micro, $SUBNET_GO)"
log "------------------------------------------------------------------------------------------------------------------------------------"

log "Building Go binary for Linux ARM64..."
cd "$GO_DIR"
GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o go-server-linux main.go
log "  Built: $(du -sh go-server-linux | cut -f1)"
cd "$ROOT_DIR"

GO_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ARM64" --instance-type t4g.micro \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_GO" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nids-go-server}]" \
  --query 'Instances[0].InstanceId' --output text --region "$REGION")
log "  Go server instance: $GO_INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$GO_INSTANCE_ID" --region "$REGION"

GO_PRIVATE_IP=$(aws ec2 describe-instances --instance-ids "$GO_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region "$REGION")
GO_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$GO_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region "$REGION")
log "  Private IP: $GO_PRIVATE_IP  |  Public IP: $GO_PUBLIC_IP"

wait_for_ssh "$GO_PUBLIC_IP"

log "Deploying Go server binary..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    "$GO_DIR/go-server-linux" ec2-user@"$GO_PUBLIC_IP":/tmp/go-server

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$GO_PUBLIC_IP" 'bash -s' <<'REMOTE'
sudo mv /tmp/go-server /usr/local/bin/go-server
sudo chmod +x /usr/local/bin/go-server
sudo bash -c 'cat > /etc/systemd/system/go-server.service' <<SVC
[Unit]
Description=Go Target Server
After=network.target
[Service]
Type=simple
ExecStart=/usr/local/bin/go-server
Environment=PORT=8080
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
SVC
sudo systemctl daemon-reload
sudo systemctl enable --now go-server
echo "Go server status: $(sudo systemctl is-active go-server)"
REMOTE
log "  --- Go server running at http://$GO_PRIVATE_IP:8080"

# ============================================================
#  STEP 2 --- XGBoost SageMaker Endpoint (ml.m5.large)
# ============================================================
log ""
log "------------------------------------------------------------------------------------------------------------------------------------"
log "  STEP 2: XGBoost SageMaker Endpoint"
log "------------------------------------------------------------------------------------------------------------------------------------"

# Pass all subnet IDs for multi-AZ SageMaker VPC config (optional but covers every subnet)
ALL_SUBNETS_CSV=$(IFS=,; echo "${SUBNETS[*]}")
log "  Subnets for SageMaker config: $ALL_SUBNETS_CSV"

cd "$DEPLOY_DIR"
python deploy_xgboost.py \
  --model-path "$MODELS_DIR/xgb_classifier.onnx" \
  --label-map-path "$MODELS_DIR/xgb_label_map.json" \
  --role "$ROLE_ARN" \
  --bucket "$BUCKET" \
  --skip-test
cd "$ROOT_DIR"
log "  --- XGBoost endpoint: $XGBOOST_ENDPOINT"

# ============================================================
#  STEP 3 --- Rust NIDS (t2.small, private IP of Go server)
# ============================================================
log ""
log "------------------------------------------------------------------------------------------------------------------------------------"
log "  STEP 3: Rust NIDS (t2.small, $SUBNET_NIDS)"
log "------------------------------------------------------------------------------------------------------------------------------------"
log "  Go server URL (private): http://$GO_PRIVATE_IP:8080"

# User-data: install Rust toolchain and build tools only
NIDS_USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex
export HOME=/root

# System deps
dnf install -y gcc openssl-devel pkg-config

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
source /root/.cargo/env

echo "Rust installed: $(rustc --version)"
echo "READY" > /tmp/rust-ready
USERDATA
)

NIDS_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_X86" --instance-type t2.small \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_NIDS" \
  --iam-instance-profile "Name=LabInstanceProfile" \
  --user-data "$NIDS_USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nids-rust-engine}]" \
  --query 'Instances[0].InstanceId' --output text --region "$REGION")
log "  Rust NIDS instance: $NIDS_INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$NIDS_INSTANCE_ID" --region "$REGION"

NIDS_PRIVATE_IP=$(aws ec2 describe-instances --instance-ids "$NIDS_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text --region "$REGION")
NIDS_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$NIDS_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region "$REGION")
log "  Private IP: $NIDS_PRIVATE_IP  |  Public IP: $NIDS_PUBLIC_IP"

wait_for_ssh "$NIDS_PUBLIC_IP"

log "Waiting for Rust install to complete on instance..."
for i in $(seq 1 30); do
  STATUS=$(ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$NIDS_PUBLIC_IP" \
    'cat /tmp/rust-ready 2>/dev/null || echo "waiting"' 2>/dev/null || echo "waiting")
  [ "$STATUS" = "READY" ] && break
  echo "  Rust install in progress... ($i/30)"; sleep 10
done

log "Syncing Rust source code and models..."
rsync -az -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
  --exclude target --exclude .git \
  "$NIDS_DIR/" ec2-user@"$NIDS_PUBLIC_IP":/home/ec2-user/ai_nids_rust/

log "Building Rust NIDS (this takes 3-5 minutes)..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$NIDS_PUBLIC_IP" 'bash -s' <<REMOTE
export HOME=/root
source /root/.cargo/env
cd /home/ec2-user/ai_nids_rust
cargo build --release 2>&1 | tail -5
echo "Build exit: \$?"
REMOTE

log "Installing Rust NIDS as systemd service..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$NIDS_PUBLIC_IP" \
  "sudo bash -c 'cat > /etc/systemd/system/nids.service'" <<EOF
[Unit]
Description=AI NIDS - Rust Engine
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/ai_nids_rust
ExecStart=/root/.cargo/bin/cargo run --release -- \\
  --autoencoder-model ./models/autoencoder.onnx \\
  --xgboost-endpoint $XGBOOST_ENDPOINT \\
  --go-server-url http://$GO_PRIVATE_IP:8080 \\
  --production-data-dir ./data
Environment=HOME=/root
Environment=AWS_DEFAULT_REGION=$REGION
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$NIDS_PUBLIC_IP" \
  'sudo systemctl daemon-reload && sudo systemctl enable nids'

log "  --- Rust NIDS deployed (service enabled, start via: sudo systemctl start nids)"

# ============================================================
#  SUMMARY
# ============================================================
log ""
log "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
log "  DEPLOYMENT COMPLETE"
log "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
log "  Lambda Shutdown"
log "    Function : $LAMBDA_FUNCTION"
log "    Trigger  : automatic (invoked by Rust NIDS on malicious detection)"
log ""
log "  Go Server"
log "    Instance : $GO_INSTANCE_ID (t4g.micro)"
log "    Subnet   : $SUBNET_GO"
log "    Private  : http://$GO_PRIVATE_IP:8080"
log "    Test     : curl http://$GO_PUBLIC_IP:8080/health"
log ""
log "  XGBoost SageMaker"
log "    Endpoint : $XGBOOST_ENDPOINT (ml.m5.large)"
log "    Subnets  : $ALL_SUBNETS_CSV"
log ""
log "  Rust NIDS"
log "    Instance : $NIDS_INSTANCE_ID (t2.small)"
log "    Subnet   : $SUBNET_NIDS"
log "    Private  : $NIDS_PRIVATE_IP"
log "    SSH      : ssh -i $KEY_PATH ec2-user@$NIDS_PUBLIC_IP"
log "    Start    : sudo systemctl start nids"
log "    Logs     : sudo journalctl -u nids -f"
log "    Trigger  : curl -X POST http://$NIDS_PUBLIC_IP:3000/trigger"
log "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

# Save deployment info
cat > "$ROOT_DIR/deployment_info.json" << DEPINFO
{
  "lambda": {
    "function_name": "$LAMBDA_FUNCTION",
    "region": "$REGION"
  },
  "go_server": {
    "instance_id": "$GO_INSTANCE_ID",
    "instance_type": "t4g.micro",
    "subnet": "$SUBNET_GO",
    "private_ip": "$GO_PRIVATE_IP",
    "public_ip": "$GO_PUBLIC_IP",
    "url": "http://$GO_PRIVATE_IP:8080"
  },
  "xgboost": {
    "endpoint": "$XGBOOST_ENDPOINT",
    "instance_type": "ml.m5.large",
    "subnets": "$ALL_SUBNETS_CSV"
  },
  "rust_nids": {
    "instance_id": "$NIDS_INSTANCE_ID",
    "instance_type": "t2.small",
    "subnet": "$SUBNET_NIDS",
    "private_ip": "$NIDS_PRIVATE_IP",
    "public_ip": "$NIDS_PUBLIC_IP",
    "trigger_url": "http://$NIDS_PUBLIC_IP:3000/trigger"
  }
}
DEPINFO
log "  Saved: deployment_info.json"
