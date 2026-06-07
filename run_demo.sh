#!/usr/bin/env bash
set -euo pipefail

# Neural RLHF Demo — Run Script
# Usage:
#   ./run_demo.sh          # connect Muse S + start demo
#   ./run_demo.sh --sim    # simulation mode (no headband)

CYAN='\033[0;36m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
DIM='\033[2m'
NC='\033[0m'

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv"

if [[ ! -d "$VENV" ]]; then
  echo -e "Run ./setup.sh first"
  exit 1
fi

# Kill leftover processes
pkill -f "neural_rlhf_server.py" 2>/dev/null || true
pkill -f "axiom_server.py" 2>/dev/null || true
sleep 0.3

SIM=""
for arg in "$@"; do
  case "$arg" in
    --sim) SIM="--sim" ;;
  esac
done

# Start frontend
echo -e "${BLUE}Starting frontend...${NC}"
cd "$DIR/frontend"
npx vite --host &
VITE_PID=$!
cd "$DIR"
sleep 2

# Start backend
echo ""
if [[ -n "$SIM" ]]; then
  echo -e "${BLUE}Starting Neural RLHF server (SIMULATION)...${NC}"
else
  echo -e "${BLUE}Starting Neural RLHF server (LIVE Muse S)...${NC}"
  echo -e "${DIM}  Make sure your Muse S is powered on${NC}"
fi
"$VENV/bin/python3" "$DIR/backend/neural_rlhf_server.py" $SIM &
SERVER_PID=$!

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          NEURAL RLHF · YOUR BRAIN AS REWARD            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Dashboard:  ${GREEN}http://localhost:5173${NC}"
echo -e "  WebSocket:  ${DIM}ws://127.0.0.1:8080${NC}"
echo ""
echo -e "  ${DIM}Press Ctrl+C to stop${NC}"
echo ""

cleanup() {
  echo ""
  echo "Shutting down..."
  kill $VITE_PID 2>/dev/null || true
  kill $SERVER_PID 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

wait $SERVER_PID
