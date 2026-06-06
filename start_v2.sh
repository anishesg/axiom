#!/bin/bash
# AXIOM v2 — Brain-Computer Interface (Research Version)
# Usage: ./start_v2.sh [--sim] [--llm] [--test]

cd "$(dirname "$0")"

source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null

mkdir -p backend/data

ARGS=""
TEST_MODE=false
for arg in "$@"; do
    ARGS="$ARGS $arg"
    if [ "$arg" = "--test" ]; then
        TEST_MODE=true
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
    echo "[0/2] Starting test site on http://localhost:8080..."
    cd backend/test_site
    python serve.py &
    TEST_SITE_PID=$!
    cd ../..
    sleep 1
fi

# Start frontend in background
echo "[1/2] Starting dashboard..."
cd frontend
npm run dev -- --port 5173 &
FRONTEND_PID=$!
cd ..

sleep 2

# Start backend
echo "[2/2] Starting BCI server..."
echo "  Args: $ARGS"
echo ""
cd backend
python axiom_v2.py $ARGS
cd ..

# Cleanup
kill $FRONTEND_PID 2>/dev/null
if [ -n "$TEST_SITE_PID" ]; then
    kill $TEST_SITE_PID 2>/dev/null
fi
