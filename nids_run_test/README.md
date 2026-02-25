# nids_run_test

CLI test trigger for the Rust NIDS server (`ai_nids_rust`) running on AWS.

## Quick start

```bash
cd nids_run_test
chmod +x trigger.sh

# Health check
./trigger.sh --host <PUBLIC_IP> --health

# Ping the Go server through the NIDS middleware
./trigger.sh --host <PUBLIC_IP> --ping

# Run 100 samples (all traffic types)
./trigger.sh --host <PUBLIC_IP>

# Run 500 DDoS samples
./trigger.sh --host <PUBLIC_IP> --attack-type "DDoS" --count 500
```

## Options

| Flag | Default | Description |
|---|---|---|
| `-H, --host` | `$NIDS_HOST` | Public IP / DNS of the Rust NIDS EC2 instance |
| `-p, --port` | `3000` | Rust server port |
| `-a, --attack-type` | *(all)* | Filter dataset by attack type |
| `-c, --count` | `100` | Number of samples to process (max 1000) |
| `-t, --timeout` | `300` | curl timeout in seconds |
| `--health` | — | Health check only |
| `--ping` | — | Ping Go server through NIDS |
| `-q, --quiet` | — | JSON-only output (machine readable) |

## Set host via env var

```bash
export NIDS_HOST=54.123.45.6
./trigger.sh --attack-type "PortScan" --count 200
```

## Available attack types

These match the labels in the production dataset:

```
DDoS           DoS Hulk       DoS GoldenEye  DoS slowloris
DoS Slowhttptest  PortScan    FTP-Patator    SSH-Patator
Web Attack - Brute Force      Web Attack - XSS
Web Attack - Sql Injection    Bot            Infiltration
Heartbleed     BENIGN
```

## What happens on malicious detection

```
trigger.sh  →  POST /trigger  →  Rust NIDS engine
                                    ↓  (if attack detected)
                              Lambda: nids-vm-shutdown
                                    ↓
                              EC2 StopInstances (nids-go-server)
```

The output summary will clearly show `Lambda: YES` if the Go server shutdown was triggered.
