# Axiom EEG Project - Setup Guide

This guide explains how to set up and run the Axiom EEG project, which consists of three main components:

1. **Eleven Server** (Python/FastAPI) - EEG processing and WebSocket API
2. **Frontend** (React/Vite) - Dashboard and calibration UI
3. **Muse Headband** - Bluetooth LE EEG device

## Prerequisites

- Python 3.12+
- Node.js 18+
- npm or yarn
- Muse S headband (or run in simulation mode)

## Quick Start

### 1. Setup Python Environment

```bash
cd /Users/krishmalik/Documents/eeg-project/axiom

# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install brainflow numpy scipy fastapi uvicorn websockets
```

### 2. Start the Eleven Server (Backend)

```bash
# From the axiom directory, with venv activated
source .venv/bin/activate
python -m uvicorn eleven.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at:
- **REST API**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/ws
- **Health check**: http://localhost:8000/health

### 3. Start the Frontend

```bash
# In a new terminal
cd /Users/krishmalik/Documents/eeg-project/axiom/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will be available at: http://localhost:5173

## Connecting the Muse Headband

1. Turn on your Muse S headband
2. Open the dashboard at http://localhost:5173
3. Click **Connect** to pair via Bluetooth
4. Wait for connection (may take 10-30 seconds for BLE discovery)
5. Once connected, you'll see live EEG waveforms

### Troubleshooting Bluetooth Connection

If the connection times out or fails:

```bash
# Kill stale BrainFlow processes
pkill -9 -f brainflow

# Kill the server
pkill -9 -f "uvicorn.*eleven"

# Restart the server
source .venv/bin/activate
python -m uvicorn eleven.server:app --host 0.0.0.0 --port 8000 --reload
```

## Running in Simulation Mode

If you don't have a Muse headband, the frontend can connect in simulation mode for testing:

1. Open the dashboard
2. Click **Connect** - it will use simulated EEG data

## Project Structure

```
axiom/
├── eleven/                 # Eleven server (Python/FastAPI)
│   ├── server.py          # Main WebSocket/REST API server
│   ├── artifacts.py       # Calibration and artifact detection
│   ├── preprocessing.py   # EEG signal preprocessing
│   ├── features.py        # Feature extraction
│   ├── acquisition.py     # Muse Bluetooth acquisition
│   └── user_profile.py    # User profile management
│
├── frontend/              # React/Vite frontend
│   ├── src/
│   │   ├── pages/         # Page components (Dashboard, Calibration)
│   │   ├── components/    # Reusable UI components
│   │   ├── hooks/         # Custom React hooks (useElevenSocket)
│   │   └── stores/        # Zustand state stores
│   └── package.json
│
├── .venv/                 # Python virtual environment
├── requirements.txt       # Python dependencies
└── SETUP.md              # This file
```

## API Endpoints

### REST API (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/session/state` | GET | Current session state |
| `/session/connect` | POST | Connect to Muse headband |
| `/session/disconnect` | POST | Disconnect from Muse |
| `/session/start-streaming` | POST | Start EEG streaming |
| `/calibration/enhanced/start` | POST | Start enhanced calibration |
| `/calibration/enhanced/step` | POST | Start a calibration step |
| `/calibration/enhanced/step/end` | POST | End current calibration step |

### WebSocket Events (ws://localhost:8000/ws)

| Event | Direction | Description |
|-------|-----------|-------------|
| `state_change` | Server→Client | Connection state updates |
| `eeg` | Server→Client | Real-time EEG data (4Hz) |
| `attention` | Server→Client | Attention metrics |
| `heartbeat` | Server→Client | Keep-alive pings |

## Calibration Flow

1. **Relax Baseline** (10s) - Eyes closed, relaxed state
2. **Focus Baseline** (8s) - Concentrated focus on a point
3. **Natural Blinks** (20s) - Normal blinking pattern
4. **Deliberate Blinks** (30s) - Firm blinks on command
5. **Jaw Clenches** (30s) - Jaw clenches on command

## Common Issues

### "Address already in use" error
```bash
lsof -ti:8000 | xargs kill -9
```

### EEG waveform is flat
- Check Muse headband is properly worn (all electrodes touching scalp)
- Apply the connectivity fix (kill stale processes, restart server)

### "Invalid step: baseline" error
- Frontend/backend step IDs were mismatched - this should be fixed now
