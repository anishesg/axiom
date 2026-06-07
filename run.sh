#!/usr/bin/env bash
set -euo pipefail

# Axiom — Run Script
# Usage:
#   ./run.sh              # full system (Muse S + overlay + dashboard)
#   ./run.sh --sim        # simulated EEG, no headband needed
#   ./run.sh --no-overlay # skip transparent overlay
#   ./run.sh --llm        # enable AWS Bedrock LLM self-improvement

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv"

if [[ ! -d "$VENV" ]]; then
  echo -e "${RED}Run ./setup.sh first${NC}"
  exit 1
fi

# Kill any leftover processes
pkill -f "axiom_server.py" 2>/dev/null || true
pkill -f "axiom_simulation.py" 2>/dev/null || true
pkill -f "overlay.py" 2>/dev/null || true
sleep 0.5

SIM=false
SERVER_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --sim) SIM=true ;;
    --no-overlay) SERVER_ARGS="$SERVER_ARGS --no-overlay" ;;
    --llm) SERVER_ARGS="$SERVER_ARGS --llm" ;;
    --gaze) SERVER_ARGS="$SERVER_ARGS --gaze" ;;
  esac
done

# Start frontend
echo -e "${BLUE}Starting frontend...${NC}"
cd "$DIR/frontend"
npx vite --host &
VITE_PID=$!
cd "$DIR"

# Wait for Vite
sleep 2

# Start backend
echo ""
if $SIM; then
  echo -e "${BLUE}Starting Axiom in SIMULATION mode...${NC}"
  echo "  (no Muse S needed — synthetic EEG)"
  "$VENV/bin/python3" "$DIR/backend/axiom_simulation.py" $SERVER_ARGS &
else
  echo -e "${BLUE}Starting Axiom — connecting to Muse S...${NC}"
  echo "  Make sure your Muse S is powered on"
  "$VENV/bin/python3" "$DIR/backend/axiom_server.py" $SERVER_ARGS &
fi
SERVER_PID=$!

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                AXIOM · RUNNING                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Dashboard:  http://localhost:5173"
echo "  WebSocket:  ws://127.0.0.1:8080"
echo ""
echo "  Press Ctrl+C to stop everything"
echo ""

cleanup() {
  echo ""
  echo -e "${BLUE}Shutting down...${NC}"
  kill $VITE_PID 2>/dev/null || true
  kill $SERVER_PID 2>/dev/null || true
  pkill -f "overlay.py" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

wait $SERVER_PID
