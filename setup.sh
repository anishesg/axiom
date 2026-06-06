#!/usr/bin/env bash
set -euo pipefail

# Axiom — Brain-Computer Interface Setup (macOS only)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                 AXIOM · SETUP                           ║"
echo "║        Brain-Computer Interface for macOS               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
  echo -e "${RED}Error: Axiom requires macOS (pyobjc, CoreGraphics, NSWindow).${NC}"
  exit 1
fi

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}Error: python3 not found. Install with: brew install python${NC}"
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MINOR" -lt 10 ]]; then
  echo -e "${RED}Error: Python 3.10+ required (found $PY_VERSION). brew install python${NC}"
  exit 1
fi
echo -e "${BLUE}[1/5]${NC} Python $PY_VERSION found"

# Check Node.js
if ! command -v node &>/dev/null; then
  echo -e "${RED}Error: node not found. Install with: brew install node${NC}"
  exit 1
fi
echo -e "${BLUE}[2/5]${NC} Node $(node -v) found"

# Python venv + deps
echo -e "${BLUE}[3/5]${NC} Setting up Python environment..."
if [[ ! -d "venv" ]]; then
  python3 -m venv venv
  echo "  Created venv/"
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Python dependencies installed"

# Frontend deps
echo -e "${BLUE}[4/5]${NC} Setting up frontend..."
cd frontend
npm install --silent 2>/dev/null
cd ..
echo "  Frontend dependencies installed"

# Bluetooth permissions reminder
echo -e "${BLUE}[5/5]${NC} Permissions check"
echo ""
echo -e "${YELLOW}  macOS permissions needed (System Settings > Privacy & Security):${NC}"
echo "    - Bluetooth    → allow Terminal / your terminal app"
echo "    - Accessibility → allow Terminal (for screen reading)"
echo "    - Input Monitoring → allow Terminal (for cursor tracking)"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "  Quick start:"
echo "    ./run.sh              # connect Muse S + start everything"
echo "    ./run.sh --sim        # run with simulated EEG (no headband)"
echo "    ./run.sh --no-overlay # skip the transparent overlay"
echo ""
echo "  Then open http://localhost:5173 in your browser."
echo ""
