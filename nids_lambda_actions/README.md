# nids_lambda_actions

AWS Lambda function invoked by the Rust NIDS engine (`ai_nids_rust`) when malicious network traffic is detected. Stops the Go server EC2 instance to isolate the compromised endpoint.

## How it fits in the pipeline

```
NIDS Rust Engine
  └─ engine/lambda.rs: LambdaTrigger::trigger_shutdown()
       └─ aws-sdk-lambda: Invokes "nids-vm-shutdown"
            └─ lambda_function.py
                 └─ boto3 ec2.stop_instances(InstanceId=<nids-go-server>)
```

The Rust engine calls Lambda **on the first malicious detection** per simulation run, sending this payload:

```json
{
  "attack_type": "DoS Hulk",
  "confidence":  0.9821,
  "mse_error":   0.0712,
  "timestamp":   "2026-02-24T18:00:00Z"
}
```

The Lambda responds with:

```json
{
  "statusCode": 200,
  "body": "{\"action\":\"stopped\",\"instance_id\":\"i-0...\",\"previous_state\":\"running\",\"current_state\":\"stopping\", ...}"
}
```

## Files

| File | Purpose |
|---|---|
| `lambda_function.py` | Lambda handler — finds and stops the Go server instance |
| `deploy.sh` | Packages + deploys (or updates) the Lambda function |
| `test_local.py` | Local dry-run test without touching AWS |
| `README.md` | This file |

## Deploy

```bash
# Set your execution role if different from the default
export LAMBDA_ROLE_ARN="arn:aws:iam::<account>:role/LabRole"

chmod +x deploy.sh
./deploy.sh
```

The script will **create** the function on first run and **update** it on subsequent runs.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `INSTANCE_TAG_KEY` | `Name` | EC2 tag key to locate the Go server |
| `INSTANCE_TAG_VAL` | `nids-go-server` | EC2 tag value to locate the Go server |
| `DRY_RUN` | `false` | Set to `true` to simulate without stopping |

## Test locally (no AWS required)

```bash
cd nids_lambda_actions
DRY_RUN=true python3 test_local.py
```

## Test on AWS (after deploy)

```bash
aws lambda invoke \
  --function-name nids-vm-shutdown \
  --payload '{"attack_type":"DDoS","confidence":0.99,"mse_error":0.08,"timestamp":"2026-02-24T18:00:00Z"}' \
  --cli-binary-format raw-in-base64-out \
  --region us-east-1 \
  response.json && cat response.json
```

## IAM Permissions

The Lambda execution role needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "ec2:DescribeInstances",
    "ec2:StopInstances"
  ],
  "Resource": "*"
}
```

`LabRole` in the AWS Academy environment already has these permissions.
