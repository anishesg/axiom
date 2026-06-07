# Eleven

**EEG-based communication platform for accessibility.**

Eleven enables people to communicate using a Muse S EEG headband, translating brain signals into discrete messages.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -e .
```

### 2. Test with Simulated Data

```bash
python -m eleven.cli test
```

This runs the full pipeline with synthetic EEG data to verify everything works.

### 3. Test Real Muse Connection

```bash
python -m eleven.cli connect
```

Ensure your Muse S is turned on and not connected to another device.

### 4. Run Calibration

```bash
python -m eleven.cli calibrate
```

Follow the prompts to calibrate the system for your unique brain patterns.

### 5. Start the Server

```bash
python -m eleven.cli server
```

The server runs at `http://localhost:8000` with WebSocket at `ws://localhost:8000/ws`.

## Architecture

```
Muse S → BrainFlow → Preprocessing → Artifact Detection → WebSocket → React UI
                                   ↓
                            VQ Encoder → Intent Classification
```

### Control Signals (95%+ accuracy)

| Signal | Action | Detection |
|--------|--------|-----------|
| Double blink | YES | Frontal amplitude (AF7/AF8) |
| Triple blink | NO | Frontal amplitude (AF7/AF8) |
| Jaw clench | SELECT | EMG on all channels |
| Long jaw clench | CANCEL | EMG duration |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/session/connect` | POST | Connect to Muse |
| `/session/start` | POST | Start streaming |
| `/session/stop` | POST | Stop streaming |
| `/session/disconnect` | POST | Disconnect |
| `/ws` | WebSocket | Real-time events |

### WebSocket Events

```json
{"type": "control_signal", "data": {"signal": "double_blink"}}
{"type": "attention", "data": {"focus": 0.7, "relaxation": 0.3}}
{"type": "heartbeat", "data": {"state": "streaming"}}
```

## Project Structure

```
eleven/
├── backend/
│   ├── eleven/
│   │   ├── acquisition.py    # Muse S connection
│   │   ├── preprocessing.py  # Signal filtering
│   │   ├── artifacts.py      # Blink/clench detection
│   │   ├── features.py       # Band power extraction
│   │   ├── vq_encoder.py     # Vector quantization
│   │   ├── server.py         # FastAPI + WebSocket
│   │   └── cli.py            # Command-line interface
│   └── pyproject.toml
├── frontend/                  # React app (TODO)
└── README.md
```

## Realistic Expectations

This system is designed for **deliberate artifact control** (blinks, clenches), not mind-reading.

| What Works | What Doesn't |
|------------|--------------|
| Blink patterns → YES/NO | Imagined speech |
| Jaw clench → SELECT | Motor imagery (no C3/C4 coverage) |
| Alpha waves → relaxation state | Full P300 speller |
| Attention metrics | SSVEP (no occipital coverage) |

Expected speed: **5-15 selections per minute** with trained user.

## Hardware Requirements

- **Muse S** (or Muse 2) headband
- Computer with Bluetooth
- macOS 12.3+ / Windows 10+ / Linux with bluez

## License

MIT
