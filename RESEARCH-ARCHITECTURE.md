# AXIOM v2: Research Architecture

## What Makes This Novel

**No published paper combines all of these in one system:**
1. Universal web element understanding (accessibility tree + visual grounding)
2. Consumer EEG with foundation model features (SingLEM) + latent state estimation (TDE-HMM)
3. Contextual bandit action selection (NeuralUCB) with ErrP reward signal
4. Webcam eye tracking with passive intent confirmation (SPN)
5. LLM for async planning/disambiguation
6. Progressive commitment UI with Bayesian intent estimation
7. Co-adaptive learning (both system and user adapt over time)

**Closest systems and why we're different:**
- NeuroChat (MIT, 2025): EEG + LLM but only adjusts text complexity, no browser control
- NeuroGaze (Frontiers, 2025): EEG + eye tracking but VR-only, P300-based (active, slow)
- WebArena agents: LLM browser control but no neural input, 2-7s latency
- UCLA AI Co-Pilot (Nature MI, 2025): Shared autonomy BCI but no web browsing, no gaze

---

## Architecture: 5-Layer Pipeline

### Layer 1: Sensing (<50ms, continuous)
```
Muse S EEG (256Hz) ──→ BrainFlow ring buffer
Webcam (30Hz)      ──→ MediaPipe FaceLandmarker → gaze (x,y)
Browser             ──→ Chrome CDP → Accessibility Tree
```

### Layer 2: Feature Extraction (<100ms)
```
EEG channels ──→ SingLEM (per-channel foundation model features)
              ──→ TDE-HMM (latent brain state at 100ms resolution)
              ──→ ErrP detector (frontal negativity at AF7/AF8)
              ──→ EMG detector (jaw clench at TP9/TP10)

Gaze ──→ I-DT fixation detector
     ──→ Spatial lookup against accessibility tree elements
     ──→ Fixation duration + saccade velocity

Page ──→ Accessibility tree element list with roles/labels
     ──→ Element bounding boxes
     ──→ Page context (URL, domain, visible text)
```

### Layer 3: Intent Estimation (<200ms, Bayesian)
```
P(intent | gaze, EEG, context) updated every frame:
  - Gaze fixation duration → likelihood of selection
  - SPN amplitude (AF7/AF8) → passive intent confirmation
  - Engagement index β/(α+θ) → action readiness
  - FAA ln(α_AF8) - ln(α_AF7) → approach/withdrawal
  - TDE-HMM state → {browsing, searching, intending, resting, error}
  - LLM prediction → prior over likely next actions (async)

Fusion: Late fusion with contribution-guided reweighting
  - When EEG noisy (high artifact rate): upweight gaze + context
  - When gaze ambiguous (dense UI): upweight EEG
  - Dynamic weighting learned from ErrP feedback
```

### Layer 4: Action Selection (NeuralUCB Contextual Bandit)
```
Context vector = [SingLEM features, HMM state, element type, 
                  gaze duration, engagement, FAA, page domain]
                  
Actions = {click, scroll_down, scroll_up, navigate_back, 
           open_link, type_text, switch_tab, close_tab, noop}

Reward = 1 - P(ErrP) within 500ms after action
         (ErrP at AF7/AF8 → negative reward → adjust policy)

Online update after every action via NeuralUCB
```

### Layer 5: Progressive Commitment UI
```
Intent probability → visual feedback:
  0.0 - 0.3: No visible change
  0.3 - 0.5: Subtle glow around gaze target element
  0.5 - 0.7: Stronger highlight + element label tooltip
  0.7 - 0.9: Bright highlight + pre-load linked content
  0.9 - 1.0: Action executes with smooth animation

Error recovery:
  ErrP detected → animated undo + show top-3 alternatives
  User selects alternative → positive reward for that action
```

---

## Key Upgrades from Stable Version

| Component | Stable (v1) | Research (v2) |
|-----------|------------|---------------|
| EEG features | Band power ratios | SingLEM foundation model |
| Brain state | Simple thresholds | TDE-HMM (5 latent states) |
| Intent detection | RandomForest classifier | NeuralUCB contextual bandit |
| Intent confirmation | Dwell time + jaw clench | SPN (passive, zero-effort) |
| Reward signal | Manual calibration labels | ErrP auto-detection |
| Website support | Gmail only | Any site via accessibility tree |
| Element understanding | Site-specific JS scanning | Chrome CDP + TinyClick fallback |
| Action vocabulary | 6 Gmail-specific actions | 8 universal primitives |
| Calibration | Per-session mandatory | EDAPT continual adaptation |
| Transfer learning | None | MAML meta-learning |
| UI feedback | Binary (act/don't act) | Progressive commitment gradient |
| LLM role | Draft replies | Async planning + disambiguation |
| Fusion | None (EEG only) | Late fusion (EEG + gaze + context) |

---

## Implementation Priority (for hackathon)

### Phase 1: Universal Browser Control
- Replace Gmail-specific JS scanning with Chrome Accessibility Tree (CDP)
- Implement universal action vocabulary (8 primitives)
- Test on Gmail, Twitter/X, Reddit, Wikipedia, YouTube

### Phase 2: Better Neural Pipeline
- Integrate EEGNet (2.6K params) as immediate upgrade over band powers
- Implement TDE-HMM for latent state estimation
- Add ErrP detector for frontal channels
- Wire ErrP → reward → NeuralUCB online learning

### Phase 3: Progressive Commitment UI
- Bayesian intent accumulator replacing binary thresholds
- Visual feedback gradient (glow/highlight/execute)
- Smooth animations for action execution and error recovery

### Phase 4: Async LLM Planning
- LLM predicts likely next actions from page context
- Pre-highlights predicted targets
- Handles disambiguation for complex pages

---

## Key Papers to Cite

1. SingLEM (arXiv 2509.17920) - Single-channel EEG foundation model
2. EDAPT (arXiv 2508.10474) - Calibration-free BCI adaptation
3. NeuroGaze (Frontiers 2025) - Hybrid EEG + eye tracking
4. SPN for target selection (CHI 2024) - Passive intent from EEG
5. NeuralUCB for BCI (Fidencio, Frontiers 2025) - Contextual bandits + ErrP
6. TDE-HMM (Frontiers Systems Neuroscience 2025) - Fast brain state estimation
7. AgentOccam (arXiv 2410.13825) - Minimal action vocabulary for web agents
8. BrowserGym (NeurIPS 2024) - Standardized web agent abstraction
9. MTREE-Net (Information Fusion 2025) - EEG + eye movement fusion
10. ZIA (arXiv 2502.16124) - Zero-Input AI framework
