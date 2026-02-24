"""
nids-vm-shutdown Lambda Function
=================================
Invoked by the Rust NIDS (ai_nids_rust) via engine/lambda.rs when the system
classifies a network packet as malicious.

Expected invocation payload (from AttackInfo in Rust):
{
    "attack_type": "DDoS",
    "confidence":  0.9821,
    "mse_error":   0.0712,
    "timestamp":   "2026-02-24T18:00:00Z"
}

Actions taken:
1. Locate the EC2 instance tagged Name=nids-go-server.
2. Stop (not terminate) the instance immediately.
3. Publish a CloudWatch Log with the attack details.
4. Return a structured response so the Rust caller can log the outcome.

Permissions required (attach to Lambda execution role):
  - ec2:DescribeInstances
  - ec2:StopInstances
  - logs:CreateLogGroup (auto-created by Lambda runtime)
  - logs:CreateLogStream
  - logs:PutLogEvents

Environment variables (set in deploy.sh or console):
  AWS_REGION        - e.g. us-east-1  (usually auto-set by Lambda runtime)
  INSTANCE_TAG_KEY  - Tag key to find the Go server  (default: Name)
  INSTANCE_TAG_VAL  - Tag value to find the Go server (default: nids-go-server)
  DRY_RUN           - Set to "true" to simulate without actually stopping the instance
"""

import json
import logging
import os
import boto3
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION           = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TAG_KEY = os.environ.get("INSTANCE_TAG_KEY", "Name")
INSTANCE_TAG_VAL = os.environ.get("INSTANCE_TAG_VAL", "nids-go-server")
DRY_RUN          = os.environ.get("DRY_RUN", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_instance_id(ec2_client) -> str | None:
    """
    Returns the first *running* instance ID tagged with our Go-server tag.
    Returns None if no match is found.
    """
    resp = ec2_client.describe_instances(
        Filters=[
            {"Name": f"tag:{INSTANCE_TAG_KEY}", "Values": [INSTANCE_TAG_VAL]},
            {"Name": "instance-state-name",     "Values": ["running", "pending"]},
        ]
    )

    for reservation in resp.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            return instance["InstanceId"]

    return None


def stop_instance(ec2_client, instance_id: str, dry_run: bool = False) -> dict:
    """
    Issues an EC2 StopInstances call and returns the state transition dict.
    If dry_run=True, performs an IAM permission check only (no actual stop).
    """
    try:
        resp = ec2_client.stop_instances(
            InstanceIds=[instance_id],
            DryRun=dry_run,
        )
        return resp.get("StoppingInstances", [{}])[0]
    except ec2_client.exceptions.ClientError as e:
        # DryRun raises DryRunOperation when permissions are OK
        if e.response["Error"]["Code"] == "DryRunOperation":
            logger.info("DRY-RUN: EC2 StopInstances permission confirmed.")
            return {"InstanceId": instance_id, "CurrentState": {"Name": "dry-run-ok"}}
        raise


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    """
    Main Lambda entry point.

    The Rust NIDS sends 'event' as a raw JSON body (not wrapped in
    API Gateway).  We parse the attack details for logging then shut
    down the Go server EC2 instance.
    """
    invocation_time = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("nids-vm-shutdown Lambda invoked at %s", invocation_time)
    logger.info("Raw event: %s", json.dumps(event))
    logger.info("=" * 60)

    # ── Parse attack payload ────────────────────────────────────────
    attack_type = event.get("attack_type", "UNKNOWN")
    confidence  = event.get("confidence",  0.0)
    mse_error   = event.get("mse_error",   0.0)
    attack_ts   = event.get("timestamp",   invocation_time)

    logger.warning(
        "ATTACK DETECTED: type=%s  confidence=%.4f  mse=%.6f  src_timestamp=%s",
        attack_type, confidence, mse_error, attack_ts,
    )

    # ── Find the Go server instance ─────────────────────────────────
    ec2 = boto3.client("ec2", region_name=REGION)

    instance_id = find_instance_id(ec2)
    if not instance_id:
        msg = (
            f"No running EC2 instance found with tag "
            f"{INSTANCE_TAG_KEY}={INSTANCE_TAG_VAL}. "
            "Nothing stopped."
        )
        logger.warning(msg)
        return _response(200, {
            "action":      "no-op",
            "reason":      msg,
            "attack_type": attack_type,
            "timestamp":   invocation_time,
        })

    logger.warning(
        "Stopping Go-server instance %s (DRY_RUN=%s)...",
        instance_id, DRY_RUN,
    )

    # ── Stop the instance ───────────────────────────────────────────
    state = stop_instance(ec2, instance_id, dry_run=DRY_RUN)

    prev_state = state.get("PreviousState", {}).get("Name", "unknown")
    curr_state = state.get("CurrentState",  {}).get("Name", "unknown")

    result = {
        "action":        "dry-run" if DRY_RUN else "stopped",
        "instance_id":   instance_id,
        "previous_state": prev_state,
        "current_state":  curr_state,
        "attack_type":   attack_type,
        "confidence":    confidence,
        "mse_error":     mse_error,
        "attack_timestamp": attack_ts,
        "lambda_timestamp": invocation_time,
    }

    logger.info("Instance state transition: %s -> %s", prev_state, curr_state)
    logger.info("Shutdown result: %s", json.dumps(result))
    logger.info("=" * 60)

    return _response(200, result)


# ---------------------------------------------------------------------------
# Response helper
# ---------------------------------------------------------------------------

def _response(status_code: int, body: dict) -> dict:
    """Return a Lambda proxy-compatible response dict."""
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
    }
