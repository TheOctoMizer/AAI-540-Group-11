#!/bin/bash
# deploy_ec2.sh — Build Go binary locally, SCP to a t2.micro EC2, run via systemd.
set -euo pipefail

REGION="us-east-1"
AMI="ami-0f3caa1cf4417e51b"      # Amazon Linux 2023 x86_64
INSTANCE_TYPE="t2.micro"          # Free-tier eligible
KEY_NAME="vockey"
KEY_PATH="${HOME}/.ssh/labsuser.pem"
SG_NAME="go-server-sg"
TAG_NAME="nids-go-server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Deploying Go Target Server to EC2"
echo "============================================================"

# ── 0. Cross-compile binary locally for Linux amd64 ────────────
echo "[1/5] Building Linux binary locally..."
cd "$SCRIPT_DIR"
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o go-server-linux main.go
echo "  Built: $(du -sh go-server-linux | cut -f1) — go-server-linux"
cd -

# ── 1. Get default VPC ──────────────────────────────────────────
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")
echo "[2/5] Default VPC: $VPC_ID"

# ── 2. Create / reuse security group ───────────────────────────
SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  echo "  Creating security group ..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Go server — SSH and HTTP 8080" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text --region "$REGION")
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22   --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 8080 --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
fi
echo "  Security group: $SG_ID"

# ── 3. Minimal user-data: just create the systemd unit skeleton ─
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
mkdir -p /opt/go-server

cat > /etc/systemd/system/go-server.service << 'SVC'
[Unit]
Description=Go Target Server
After=network.target

[Service]
Type=simple
ExecStart=/opt/go-server/go-server
Environment=PORT=8080
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable go-server
USERDATA
)

# ── 4. Launch instance ──────────────────────────────────────────
echo "[3/5] Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
  --query 'Instances[0].InstanceId' \
  --output text --region "$REGION")
echo "  Instance ID: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region "$REGION")
echo "  Public IP: $PUBLIC_IP"

# ── 5. Wait for SSH, SCP the binary, start service ─────────────
echo "[4/5] Waiting for SSH to be ready..."
for i in $(seq 1 30); do
  if ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
       ec2-user@"$PUBLIC_IP" 'exit' 2>/dev/null; then
    echo "  SSH ready after ${i}×5 seconds"
    break
  fi
  echo "  Attempt $i/30 — retrying in 5s..."
  sleep 5
done

echo "[5/5] Copying binary and starting service..."
scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/go-server-linux" ec2-user@"$PUBLIC_IP":/tmp/go-server

ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ec2-user@"$PUBLIC_IP" \
  'sudo mv /tmp/go-server /opt/go-server/go-server && \
   sudo chmod +x /opt/go-server/go-server && \
   sudo systemctl start go-server && \
   sudo systemctl status go-server --no-pager'

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo "  Instance : $INSTANCE_ID ($INSTANCE_TYPE)"
echo "  Public IP: $PUBLIC_IP"
echo "  Key pair : $KEY_NAME"
echo "============================================================"
echo ""
echo "  Test:"
echo "    curl http://$PUBLIC_IP:8080/health"
echo "    curl http://$PUBLIC_IP:8080/ping"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_PATH ec2-user@$PUBLIC_IP"
echo ""
echo "  Logs:"
echo "    ssh -i $KEY_PATH ec2-user@$PUBLIC_IP 'sudo journalctl -u go-server -f'"
echo ""
echo "  Use in Rust:"
echo "    --go-server-url http://$PUBLIC_IP:8080"
echo "============================================================"
