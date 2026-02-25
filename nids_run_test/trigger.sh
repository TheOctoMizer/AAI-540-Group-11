#!/bin/bash
# ============================================================
#  trigger.sh  —  Invoke the Rust NIDS server on AWS
#
#  Sends a POST /trigger request to the ai_nids_rust engine
#  with configurable attack type and sample count.
#
#  Usage:
#    ./trigger.sh [OPTIONS]
#
#  Options:
#    -H, --host HOST          Public IP or DNS of the Rust NIDS instance
#                             (or set NIDS_HOST env var)
#    -p, --port PORT          Port the Rust server listens on (default: 3000)
#    -a, --attack-type TYPE   Filter dataset by attack type
#                             e.g. "DoS Hulk", "DDoS", "PortScan"
#                             Omit to run on all traffic.
#    -c, --count N            Number of samples to process (default: 100, max 1000)
#    -t, --timeout SECS       curl timeout in seconds (default: 300)
#        --health             Only run a health check, do not trigger simulation
#        --ping               Ping the Go server through the Rust middleware
#    -q, --quiet              Suppress banner; only print the JSON response
#    -h, --help               Show this help message
#
#  Examples:
#    # Trigger with defaults (100 random samples)
#    ./trigger.sh --host 54.123.45.6
#
#    # Simulate 500 DDoS samples
#    ./trigger.sh --host 54.123.45.6 --attack-type "DDoS" --count 500
#
#    # Just check health
#    ./trigger.sh --host 54.123.45.6 --health
#
#    # Use env var for host
#    export NIDS_HOST=54.123.45.6
#    ./trigger.sh --count 200
# ============================================================
set -euo pipefail

# ── Colours ─────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

# ── Defaults ─────────────────────────────────────────────────
HOST="${NIDS_HOST:-}"
PORT=3000
ATTACK_TYPE=""
COUNT=100
TIMEOUT=300
MODE="trigger"   # trigger | health | ping
QUIET=false

# ── Helpers ──────────────────────────────────────────────────
usage() {
  cat << 'EOF'
Usage: trigger.sh [OPTIONS]

Options:
  -H, --host HOST          Public IP or DNS of the Rust NIDS instance
                           (or set NIDS_HOST env var)
  -p, --port PORT          Port the Rust server listens on (default: 3000)
  -a, --attack-type TYPE   Filter dataset by attack type
                           e.g. "DoS Hulk", "DDoS", "PortScan"
                           Omit to run on all traffic.
  -c, --count N            Number of samples to process (default: 100, max 1000)
  -t, --timeout SECS       curl timeout in seconds (default: 300)
      --health             Only run a health check, do not trigger simulation
      --ping               Ping the Go server through the Rust middleware
  -q, --quiet              Suppress banner; only print the JSON response
  -h, --help               Show this help message

Examples:
  ./trigger.sh --host 54.123.45.6
  ./trigger.sh --host 54.123.45.6 --attack-type "DDoS" --count 500
  ./trigger.sh --host 54.123.45.6 --health
  export NIDS_HOST=54.123.45.6 && ./trigger.sh --count 200
EOF
  exit 0
}

die()  { echo -e "${RED}ERROR: $*${RST}" >&2; exit 1; }
info() { $QUIET || echo -e "${CYN}$*${RST}"; }
ok()   { $QUIET || echo -e "${GRN}$*${RST}"; }
warn() { $QUIET || echo -e "${YEL}$*${RST}"; }

banner() {
  $QUIET && return
  echo -e "${BLD}"
  echo "  ╔══════════════════════════════════════════════╗"
  echo "  ║       NIDS Rust Engine  —  Test Trigger      ║"
  echo "  ╚══════════════════════════════════════════════╝${RST}"
  echo ""
}

pretty_json() {
  # Pretty-print JSON if python3 or jq is available, else raw
  if command -v jq &>/dev/null; then
    echo "$1" | jq .
  elif command -v python3 &>/dev/null; then
    echo "$1" | python3 -m json.tool
  else
    echo "$1"
  fi
}

summarise() {
  local json="$1"
  if command -v python3 &>/dev/null; then
    python3 - "$json" << 'PYEOF'
import json, sys
try:
    d = json.loads(sys.argv[1])
    total     = d.get("total_samples", "?")
    benign    = d.get("benign_count",  "?")
    anomalous = d.get("anomalous_count", "?")
    malicious = d.get("malicious_count", "?")
    triggered = d.get("lambda_triggered", False)
    print(f"\n  Samples   : {total}")
    print(f"  Benign    : {benign}")
    print(f"  Anomalous : {anomalous}")
    print(f"  Malicious : {malicious}")
    trigger_str = "\033[0;31mYES — Go server shutdown initiated!\033[0m" if triggered else "No"
    print(f"  Lambda    : {trigger_str}")
except Exception:
    pass
PYEOF
  fi
}

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -H|--host)         HOST="$2";        shift 2 ;;
    -p|--port)         PORT="$2";        shift 2 ;;
    -a|--attack-type)  ATTACK_TYPE="$2"; shift 2 ;;
    -c|--count)        COUNT="$2";       shift 2 ;;
    -t|--timeout)      TIMEOUT="$2";     shift 2 ;;
    --health)          MODE="health";    shift   ;;
    --ping)            MODE="ping";      shift   ;;
    -q|--quiet)        QUIET=true;       shift   ;;
    -h|--help)         usage ;;
    *) die "Unknown argument: $1  (use --help for usage)" ;;
  esac
done

# ── Validate ─────────────────────────────────────────────────
[ -n "$HOST" ] || die "No host specified. Use --host <IP> or export NIDS_HOST=<IP>"

if [[ "$HOST" == http://* ]] || [[ "$HOST" == https://* ]]; then
  BASE_URL="${HOST}"
  # If host contains protocol, port might already be included or not. 
  # If it's just http://ip, we might still want to append the port if it's missing.
  # But for simplicity, if protocol is provided, we assume the user knows what they are doing.
else
  BASE_URL="http://${HOST}:${PORT}"
fi

# ── Execute ──────────────────────────────────────────────────
banner

case "$MODE" in

  # ──────────────── Health check ────────────────────────────
  health)
    info "Checking health at ${BASE_URL}/health ..."
    RESP=$(curl -sf --max-time 10 "${BASE_URL}/health" 2>&1) \
      && { ok "  Healthy!"; pretty_json "$RESP"; } \
      || die "Health check failed — is the server running?"
    ;;

  # ──────────────── Ping Go server ────────────────────────────
  ping)
    info "Pinging Go server via ${BASE_URL}/ping ..."
    RESP=$(curl -sf --max-time 10 "${BASE_URL}/ping" 2>&1) \
      && { ok "  Go server reachable!"; echo "  Response: $RESP"; } \
      || die "Ping failed — Go server may be unreachable from the NIDS instance."
    ;;

  # ──────────────── Full simulation trigger ─────────────────
  trigger)
    # Build JSON payload
    if [ -n "$ATTACK_TYPE" ]; then
      PAYLOAD="{\"attack_type\":\"${ATTACK_TYPE}\",\"count\":${COUNT}}"
    else
      PAYLOAD="{\"count\":${COUNT}}"
    fi

    info "Host        : ${HOST}:${PORT}"
    info "Samples     : ${COUNT}"
    [ -n "$ATTACK_TYPE" ] && info "Attack type : ${ATTACK_TYPE}" || info "Attack type : (all)"
    info "Timeout     : ${TIMEOUT}s"
    echo ""
    warn "Sending trigger — this may take up to ${TIMEOUT}s ..."
    echo ""

    HTTP_CODE=$(curl -s -o /tmp/nids_response.json -w "%{http_code}" \
      --max-time "$TIMEOUT" \
      -X POST "${BASE_URL}/trigger" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD") || die "curl failed — is the server reachable at ${BASE_URL}?"

    RESP=$(cat /tmp/nids_response.json)

    if [ "$HTTP_CODE" = "200" ]; then
      ok "Response (HTTP ${HTTP_CODE}):"
      $QUIET && echo "$RESP" || pretty_json "$RESP"
      $QUIET || summarise "$RESP"
    else
      warn "Response (HTTP ${HTTP_CODE}):"
      pretty_json "$RESP"
      die "Server returned HTTP ${HTTP_CODE}"
    fi
    ;;
esac

echo ""
