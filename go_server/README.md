# Go Target Server

Simple HTTP server that acts as the "target" VM in the NIDS demo.

## Endpoints

- `GET /health` - Health check
- `GET /ping` - Ping endpoint (called by NIDS middleware)

## Running

```bash
# Default port 8080
go run main.go

# Custom port
PORT=9000 go run main.go
```

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Ping
curl http://localhost:8080/ping
```

## Purpose

This server represents the target VM that will be shut down by Lambda when malicious traffic is detected by the NIDS system.
