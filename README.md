# Axiom — Brain-Computer Interface

Control your Mac with your brain. Uses a Muse S EEG headband to read engagement, focus, and relaxation — then acts on what you're looking at.

## What it does

- **Real-time EEG** — 4-channel, 256Hz from Muse S via BrainFlow
- **Brain state classification** — engagement, focus, relaxation, cognitive load, valence
- **Intent ring** — sustained focus on a target fills a ring, then confirms the action
- **Context switching** — detects when you want to switch apps from brain patterns
- **Transparent overlay** — presence indicator, intent ring, gaze trail rendered over macOS
- **Self-improvement** — LLM reviews action accuracy and adjusts thresholds (optional, via AWS Bedrock)
- **Live dashboard** — React app with EEG waveforms, brain states, agent decisions

## Requirements

- **macOS** (uses pyobjc, CoreGraphics, NSWindow)
- **Python 3.10+**
- **Node.js 18+**
- **Muse S headband** (or run in simulation mode)

## Setup

```bash
git clone https://github.com/YOUR_USER/axiom.git
cd axiom
./setup.sh
```

Grant Terminal these permissions in System Settings > Privacy & Security:
- Bluetooth
- Accessibility
- Input Monitoring

## Run

```bash
# Full system — connect Muse S headband
./run.sh

# Simulation mode — no headband needed
./run.sh --sim

# Skip the transparent overlay
./run.sh --no-overlay

# Enable LLM self-improvement (needs AWS CLI configured)
./run.sh --llm
```

Then open **http://localhost:5173**.

Calibration runs first (relax baseline + focus baseline), then the dashboard shows live EEG and brain states.

## Architecture

```
Muse S (BLE)
    │
    ▼
BrainFlow (256Hz EEG)
    │
    ▼
BrainStateEngine ──► engagement, focus, relaxation, cognitive_load, valence
    │
    ▼
AxiomAgent ──► intent ring, context switch, scroll, error detection
    │
    ├──► macOS Overlay (transparent, click-through, all Spaces)
    ├──► OS Actions (click, scroll, switch app — via pyautogui + accessibility)
    └──► WebSocket → React Dashboard
```

Input: cursor position is used as gaze proxy (webcam gaze tracking planned).

## Project structure

```
backend/
  axiom_server.py      — main server (BrainFlow + WebSocket + agent + overlay)
  axiom_simulation.py  — demo server with synthetic EEG
  brain_state.py       — EEG → cognitive state classifier
  axiom_agent.py       — decision engine (thresholds + intent ring + RL)
  os_control.py        — macOS screen reading + actions
  overlay.py           — transparent NSWindow overlay process
  overlay_ctrl.py      — overlay controller (spawns overlay subprocess)
  llm_engine.py        — AWS Bedrock integration for self-improvement
  gaze_tracker.py      — MediaPipe face mesh gaze tracking (needs webcam)
  features.py          — 29-dim RL state vector extraction
  server.py            — basic EEG streaming server (no agent)

frontend/
  src/App.tsx                    — calibration → dashboard routing
  src/components/Calibration.tsx — brain baseline calibration
  src/components/EEGDashboard.tsx — main dashboard
  src/components/BrainStatePanel.tsx — brain state bars
  src/components/ActionFeed.tsx  — agent action log + accuracy
  src/hooks/useMuseStream.ts    — WebSocket hook
```
