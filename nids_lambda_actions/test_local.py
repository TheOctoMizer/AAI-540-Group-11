"""
Local test for nids-vm-shutdown Lambda.

Runs the Lambda handler with a mock AttackInfo payload.
Set DRY_RUN=true in the environment to avoid stopping any real instance.

Usage:
    DRY_RUN=true python3 test_local.py
"""

import json
import os

# Force dry-run for local testing
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("INSTANCE_TAG_KEY", "Name")
os.environ.setdefault("INSTANCE_TAG_VAL", "nids-go-server")

import lambda_function  # noqa: E402 (import after env setup)


class MockContext:
    """Minimal stand-in for the Lambda context object."""
    function_name = "nids-vm-shutdown"
    aws_request_id = "local-test-0001"
    log_group_name = "/aws/lambda/nids-vm-shutdown"


def main():
    # Simulated AttackInfo payload from Rust
    event = {
        "attack_type": "DoS Hulk",
        "confidence":  0.9821,
        "mse_error":   0.0712,
        "timestamp":   "2026-02-24T18:00:00+00:00",
    }

    print("=" * 60)
    print("LOCAL LAMBDA TEST: nids-vm-shutdown")
    print("=" * 60)
    print("  DRY_RUN mode:", os.environ["DRY_RUN"])
    print("  Payload:", json.dumps(event, indent=2))
    print("=" * 60)

    response = lambda_function.lambda_handler(event, MockContext())

    print()
    print("=" * 60)
    print("Lambda Response:")
    print(json.dumps(response, indent=2))
    print("=" * 60)

    body = json.loads(response["body"])
    assert response["statusCode"] == 200, "Expected statusCode 200"
    print(f"\n  Action: {body.get('action')}")
    print(f"  Attack: {body.get('attack_type')}")
    print("\nTest passed.")


if __name__ == "__main__":
    main()
