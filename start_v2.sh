#!/bin/bash
# AXIOM v2 — Brain-Computer Interface (Research Version)
# Usage: ./start_v2.sh [--sim] [--llm]

cd "$(dirname "$0")"

source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null

mkdir -p backend/data

ARGS=""
for arg in "$@"; do
    ARGS="$ARGS $arg"
done

echo "═══════════════════════════════════════════"
echo "  AXIOM v2 — Universal Brain-Computer Interface"
echo "═══════════════════════════════════════════"
echo ""
echo "  Features: EEGNet + TDE-HMM + NeuralUCB + Bayesian Intent"
echo "  Works on: Any website (universal accessibility tree)"
echo ""

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
