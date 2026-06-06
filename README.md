# AXIOM v2 — Universal Brain-Computer Interface

**WeaveHacks 4 · June 6-7, 2026**

Control any website with your brain. Muse S EEG headband reads neural intent, webcam tracks your gaze, and an RL agent learns which actions you want. Look at a button, think about clicking it, and it clicks. No hands required.

```
Muse S EEG (4ch, 256Hz)  ──►  EEGNet (32-dim features)
                                    │
Webcam (gaze @ 15Hz)  ──────►  Bayesian Intent  ──►  Progressive Commitment
                                Accumulator              (glow → highlight → execute)
                                    │
                              TDE-HMM (5 brain states)
                                    │
                              NeuralUCB Bandit  ──►  Playwright Browser Action
                                    │
                              ErrP Detector  ──►  Undo if error detected
```

## Quick Start (MacBook Air)

### Prerequisites

- macOS with Bluetooth
- Python 3.10+ (`python3 --version`)
- Node.js 18+ (`node --version`)
- A webcam (built-in MacBook camera works)
- Muse S headband (or use `--sim` to skip)

### 1. Clone and Setup

```bash
git clone <repo-url> muse-brain
cd muse-brain

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install brainflow==5.22.2 websockets numpy scipy opencv-contrib-python mediapipe playwright

# Install Playwright browser (one-time)
python -m playwright install chromium

# Download MediaPipe face model (one-time)
mkdir -p backend/data
curl -L -o backend/data/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. macOS Camera Permission

The first time you run, macOS will ask for camera access. If it doesn't, or if gaze tracking fails:

1. Open **System Settings → Privacy & Security → Camera**
2. Make sure **Terminal** (or your terminal app) is toggled ON
3. If you changed the setting, restart your terminal

### 3. Run with Test Site (recommended first time)

This opens a dummy email inbox designed for easy BCI testing — large buttons, clear labels, action log.

```bash
chmod +x start_v2.sh
./start_v2.sh --sim --test
```

Flags:
- `--sim` — Simulate EEG (no Muse headband needed)
- `--test` — Launch dummy test site on localhost:8080 and navigate to it
- `--llm` — Enable LLM-assisted action decisions (needs AWS Bedrock credentials)

### 4. Open the Dashboard

Go to **http://localhost:5173** in a separate browser window. You'll see:

- **Left panel**: Neural state — EEGNet features, HMM brain states, engagement/focus
- **Center**: Commitment ring — fills up as the system becomes more confident you want to act
- **Right panel**: Action feed — what the BCI decided to do and why

### 5. Run with Real Muse S

The Bluetooth connection runs as a **separate persistent process** (`muse_bridge.py`) so you can restart the backend without losing the BT link — the hardest part of Muse development.

```bash
# Turn on your Muse S headband (hold power button until LED pulses)
# Make sure Bluetooth is on

# Option A: start_v2.sh handles everything (starts bridge automatically)
./start_v2.sh --test

# Option B: run bridge separately (recommended for development)
python backend/muse_bridge.py          # Connects to Muse, runs forever
# In another terminal:
./start_v2.sh --test                   # Detects bridge, uses it automatically
```

The bridge stays alive when you Ctrl+C the backend. Restart axiom_v2.py as many times as you want — BT stays connected.

```bash
python backend/muse_bridge.py --status  # Check bridge status
python backend/muse_bridge.py --stop    # Disconnect Muse and stop bridge
python backend/muse_bridge.py --sim     # Run bridge with simulated EEG
```

If connection fails:
- Make sure no other app (Muse Direct, Mind Monitor) is connected to it
- Turn the Muse off and on again
- Check Bluetooth is enabled in macOS settings

### 6. Gaze Calibration

When the system starts:
1. The dashboard shows a calibration screen with 9 dots
2. Look at each dot for ~2 seconds when it highlights
3. After all 9 points, calibration is saved to `backend/data/gaze_calibration.json`
4. Next run loads the saved calibration automatically (recalibrate if you move your laptop)

## How It Works

### Neural Pipeline

1. **EEGNet** (2.6K params CNN) extracts 32-dim features from raw EEG every 500ms
2. **TDE-HMM** (5 hidden states) tracks your brain state: browsing, searching, intending, resting, error
3. **Bayesian Intent Accumulator** builds up P(intent) per UI element from gaze + neural signals
4. **NeuralUCB Bandit** (7 actions) learns which action to take given brain state + element type
5. **ErrP Detector** monitors frontal EEG for error-related potentials after each action — undoes bad actions

### Progressive Commitment

The system doesn't just click instantly. It progressively commits:

| Level | P(intent) | Visual Feedback |
|-------|-----------|----------------|
| None | < 0.3 | No feedback |
| Subtle | 0.3 | Faint glow around element |
| Medium | 0.5 | Visible highlight |
| Strong | 0.7 | Bright highlight + ring filling |
| Execute | 0.9 | Action fires |

If you look away, the probability decays. If ErrP is detected after action, it undoes.

### Jaw Clench Backup

Clench your jaw firmly for an explicit "click" — detected as EMG artifact at temporal channels (TP9/TP10) with ~95% accuracy.

## Project Structure

```
muse-brain/
├── backend/
│   ├── axiom_v2.py              # Main orchestrator (6 async loops)
│   ├── muse_bridge.py           # Persistent BT process (shared memory EEG stream)
│   ├── universal_controller.py  # Playwright browser control via accessibility tree
│   ├── neural_pipeline.py       # EEGNet + HMM + UCB + Intent + ErrP
│   ├── brain_state.py           # Basic EEG band power features
│   ├── gaze_tracker.py          # MediaPipe face landmarks → gaze position
│   ├── llm_engine.py            # Optional LLM action reasoning
│   ├── test_site/
│   │   ├── index.html           # Dummy email inbox for testing
│   │   └── serve.py             # Simple HTTP server (port 8080)
│   └── data/
│       ├── face_landmarker.task # MediaPipe model (downloaded in setup)
│       └── gaze_calibration.json # Saved calibration (auto-created)
├── frontend/
│   └── src/
│       ├── App.tsx               # 3-panel neural dashboard
│       └── index.css             # Dark sci-fi theme
├── start_v2.sh                   # Launch script
└── README.md
```

## Troubleshooting

### "No module named 'brainflow'"
```bash
source venv/bin/activate  # Make sure venv is active
pip install brainflow==5.22.2
```

### "Camera not found" or gaze tracking fails
- Check System Settings → Privacy & Security → Camera → Terminal is ON
- Close any other app using the camera (Zoom, FaceTime, Photo Booth)
- Restart terminal after granting permission

### "MediaPipe face_landmarker.task not found"
```bash
curl -L -o backend/data/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

### Muse S won't connect
- Only one app can connect to it at a time — close Muse Direct / Mind Monitor
- Turn Muse off (hold 5s) then on again
- `--sim` flag works without any headband

### Playwright browser won't launch
```bash
python -m playwright install chromium
```

### Port already in use
```bash
# Kill existing processes on the ports
lsof -ti:8765 | xargs kill  # WebSocket server
lsof -ti:5173 | xargs kill  # Frontend
lsof -ti:8080 | xargs kill  # Test site
```

### Dashboard shows no data
- Make sure the backend is running (check terminal for "Axiom v2 server on ws://127.0.0.1:8765")
- Open browser console (F12) and check for WebSocket connection errors
- The frontend connects to `ws://127.0.0.1:8765` — make sure nothing else is on that port

## Test Site Usage

The test site (`--test` flag) is a self-contained email inbox with:
- 8 email rows with sender, subject, date
- Sidebar navigation (Inbox, Starred, Sent, Drafts, Trash)
- Compose, Archive, Delete, Reply buttons
- Search bar
- Action log panel showing BCI actions in real-time

All elements have explicit ARIA labels and roles for clean accessibility tree scanning. Elements are 48px+ tall for reliable gaze targeting.

Use the test site to verify:
1. **Element scanning** — check the dashboard shows all inbox elements in the right panel
2. **Gaze tracking** — move your eyes across the screen, verify the gaze dot tracks reasonably
3. **Intent accumulation** — stare at a button, watch the commitment ring fill up
4. **Action execution** — when commitment hits 0.9, the action should fire and appear in the action log

## Architecture Reference

See [RESEARCH-ARCHITECTURE.md](RESEARCH-ARCHITECTURE.md) for the full research-version architecture, literature references, and comparison with the stable Gmail-only version.
