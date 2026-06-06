# Neural RLHF — Your Brain as the Reward Model

**WeaveHacks 4 · June 6-7, 2026**

Three AI agents compete to explain topics. You read their responses wearing a Muse S EEG headband. Your brain's neural response — engagement, focus, valence — is the reward signal. Agents that resonate with your brain get reinforced. Over rounds, they converge on YOUR preferred style. No clicking, no typing. Just thinking.

## How It Works

```
Muse S EEG (256Hz)
    │
    ▼
Brain State Engine → engagement, focus, valence, cognitive load
    │
    ▼
Neural Reward Model → maps brain response to reward scores per agent
    │
    ├──► Agent Arena (3 competing agents via OpenAI)
    │       ├── The Researcher 🔬 — depth & rigor
    │       ├── The Storyteller ✨ — narratives & metaphor
    │       └── The Engineer ⚡ — practical & direct
    │
    ├──► Weave (traces full neural→agent→reward pipeline)
    └──► Redis (EEG streams, brain-state memory)
```

## Quick Start

```bash
# Setup (first time)
./setup.sh

# Run with Muse S
./run_demo.sh

# Run in simulation mode (no headband)
./run_demo.sh --sim
```

Then open **http://localhost:5173**

## Demo Flow

1. **Calibrate** — 20s baseline recording (or skip for demo)
2. **Choose a topic** — type your own or pick a preset
3. **Read** — each agent's response is shown for 12s while your brain is measured
4. **Results** — neural rewards computed, winner announced
5. **Adapt** — losing agents get feedback from your neural response
6. **Repeat** — over 3-5 rounds, agents converge on your preferences

## Requirements

- macOS with Bluetooth
- Python 3.10+
- Node.js 18+
- Muse S headband (or use `--sim` mode)
- OpenAI API key (`export OPENAI_API_KEY=...`)

## Project Structure

```
muse-brain/
├── backend/
│   ├── neural_rlhf_server.py   # Main server — EEG + agents + rewards
│   ├── brain_state.py          # EEG → cognitive state classifier
│   ├── neural_reward.py        # Brain state → reward scores
│   ├── agent_arena.py          # Multi-agent system with adaptation
│   ├── features.py             # 29-dim RL state vector extraction
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx              # Neural RLHF dashboard
│       └── index.css            # Dark theme
├── run_demo.sh                  # One-command launch
├── setup.sh                     # Environment setup
└── README.md
```

## Sponsor Integrations

- **Weave** — `@weave.op()` on agent generation, reward computation. `weave.attributes()` attaches brain state to every trace.
- **Redis** — Streams for live EEG ingestion, Pub/Sub for reward broadcasting, Vector Search for brain-state memory.
- **OpenAI** — GPT-4o-mini for agent generation via Responses API.

## Key Signals

| Signal | Source | What It Means |
|--------|--------|---------------|
| Engagement | β/(α+θ) | Locked in, interested |
| Focus | β/α | Deep processing |
| Valence | FAA (AF8-AF7 α) | Positive vs negative affect |
| Cognitive Load | θ/α | Confused, overwhelmed |
| Relaxation | α dominance | Disengaged |

## Research

- Krigolson et al. 2017 — Reward Positivity validated on Muse TP9/TP10
- Zhang et al. 2026 (arXiv:2603.16897) — EEG-LLM preference alignment, 80.68% accuracy
- Kim et al. 2025 — Neural RLHF for robot learning
