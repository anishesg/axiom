#!/bin/bash
# Gmail BCI — Launch Script
# Usage: ./start_gmail.sh [--sim] [--llm]

cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null

# Create data directory
mkdir -p backend/data

# Parse args
ARGS=""
for arg in "$@"; do
    ARGS="$ARGS $arg"
done

echo "═══════════════════════════════════════════"
echo "  AXIOM.GMAIL — Brain-Computer Interface"
echo "═══════════════════════════════════════════"
echo ""

# Start frontend in background
echo "[1/2] Starting dashboard..."
cd frontend
npm run dev -- --port 5173 &
FRONTEND_PID=$!
cd ..

# Wait for frontend
sleep 2

# Start backend
echo "[2/2] Starting BCI server..."
echo "  Args: $ARGS"
echo ""
cd backend
python gmail_bci.py $ARGS
cd ..

# Cleanup
kill $FRONTEND_PID 2>/dev/null
