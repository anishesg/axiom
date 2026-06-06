#!/bin/bash
# AXIOM v2 — Brain-Computer Interface (Research Version)
# Usage: ./start_v2.sh [--sim] [--llm] [--test]
#
# The Muse S Bluetooth connection runs as a separate persistent process
# (muse_bridge.py) so the backend can restart without losing the BT link.
#
# To run the bridge separately:
#   python backend/muse_bridge.py          # real Muse
#   python backend/muse_bridge.py --sim    # simulated
#   python backend/muse_bridge.py --status # check status
#   python backend/muse_bridge.py --stop   # stop bridge

cd "$(dirname "$0")"

source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null

mkdir -p backend/data

ARGS=""
TEST_MODE=false
SIM_MODE=false
for arg in "$@"; do
    ARGS="$ARGS $arg"
    if [ "$arg" = "--test" ]; then
        TEST_MODE=true
    fi
    if [ "$arg" = "--sim" ]; then
        SIM_MODE=true
    fi
done

echo "═══════════════════════════════════════════"
echo "  AXIOM v2 — Universal Brain-Computer Interface"
echo "═══════════════════════════════════════════"
echo ""
echo "  Features: EEGNet + TDE-HMM + NeuralUCB + Bayesian Intent"
echo "  Works on: Any website (universal accessibility tree)"
echo ""

# Start test site if --test
TEST_SITE_PID=""
if [ "$TEST_MODE" = true ]; then
    echo "[0/3] Starting test site on http://localhost:8080..."
    cd backend/test_site
    python serve.py &
    TEST_SITE_PID=$!
    cd ../..
    sleep 1
fi

# Start muse_bridge if not already running (and not in sim mode without bridge)
BRIDGE_PID=""
if [ -f /tmp/muse_bridge.pid ] && kill -0 "$(cat /tmp/muse_bridge.pid)" 2>/dev/null; then
    echo "[1/3] Muse bridge already running (pid=$(cat /tmp/muse_bridge.pid))"
else
    echo "[1/3] Starting Muse bridge..."
    cd backend
    if [ "$SIM_MODE" = true ]; then
        python muse_bridge.py --sim &
    else
        python muse_bridge.py &
    fi
    BRIDGE_PID=$!
    cd ..
    sleep 2
fi

# Start frontend in background
echo "[2/3] Starting dashboard..."
cd frontend
npm run dev -- --port 5173 &
FRONTEND_PID=$!
cd ..

sleep 2

# Start backend
echo "[3/3] Starting BCI server..."
echo "  Args: $ARGS"
echo ""
cd backend
python axiom_v2.py $ARGS
cd ..

# Cleanup — kill frontend and test site, but KEEP the bridge running
kill $FRONTEND_PID 2>/dev/null
if [ -n "$TEST_SITE_PID" ]; then
    kill $TEST_SITE_PID 2>/dev/null
fi
echo ""
echo "Backend stopped. Muse bridge is still running."
echo "  Check: python backend/muse_bridge.py --status"
echo "  Stop:  python backend/muse_bridge.py --stop"
