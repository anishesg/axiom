# Muse S × BrainFlow — Full Capability Map for RL/Agentic Systems

## What BrainFlow Gives Us (that muselsl doesn't)

### Signal Acquisition
- **Native BLE** — direct connection, no LSL middleware
- **Multi-preset streaming** — EEG (256Hz), IMU (52Hz), PPG (64Hz) simultaneously
- **Configurable presets** — p21 (EEG+IMU), p50/p51 (EEG+IMU+PPG)

### Built-in DSP (C-optimized, runs in <1ms)
- Bandpass/bandstop/highpass/lowpass (Butterworth, Chebyshev, Bessel + zero-phase variants)
- Environmental noise removal (50/60 Hz notch)
- Wavelet denoising (db1-db15, haar, sym, coif, bior families)
- FFT/IFFT, Welch PSD, band power extraction
- Rolling filters (moving average, median)
- ICA (Independent Component Analysis) — artifact removal
- CSP (Common Spatial Patterns) — spatial filtering for BCI

### Built-in ML Classifiers
- `MLModel` with concentration/relaxation metrics
- Supports custom ONNX models — train your own, deploy on-device
- Band power ratios → feature vectors ready for any ML pipeline

### Built-in Biometrics
- `DataFilter.get_heart_rate()` — HR from PPG
- `DataFilter.get_oxygen_level()` — SpO2 from PPG
- `DataFilter.get_railed_percentage()` — signal quality metric

---

## RL/Agentic Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MUSE S HEADBAND                          │
│  EEG (4ch, 256Hz) · PPG (3ch, 64Hz) · IMU (6ch, 52Hz)    │
└─────────────────┬───────────────────────────────────────────┘
                  │ BLE
┌─────────────────▼───────────────────────────────────────────┐
│              BRAINFLOW ACQUISITION LAYER                     │
│  BoardShim → prepare_session → start_stream → get_board_data│
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              FEATURE EXTRACTION PIPELINE                     │
│                                                              │
│  Raw EEG → Filter → Band Powers → Feature Vector            │
│    ├─ Delta (1-4Hz)   — deep sleep, unconscious             │
│    ├─ Theta (4-8Hz)   — meditation, creativity, drowsiness  │
│    ├─ Alpha (8-13Hz)  — relaxation, eyes closed, calm focus │
│    ├─ Beta (13-30Hz)  — active thinking, concentration      │
│    └─ Gamma (30-50Hz) — peak cognition, flow state          │
│                                                              │
│  Derived Features:                                           │
│    ├─ Alpha/Theta ratio → focus vs relaxation                │
│    ├─ Beta/Alpha ratio  → engagement level                   │
│    ├─ Frontal asymmetry (AF7 vs AF8) → emotional valence     │
│    ├─ Cross-channel coherence → cognitive integration        │
│    ├─ Heart rate variability (PPG) → stress/recovery         │
│    └─ Head movement (IMU) → restlessness, nods               │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              STATE REPRESENTATION (for RL)                    │
│                                                              │
│  s_t = [                                                     │
│    band_powers[5],        # δ, θ, α, β, γ                   │
│    band_ratios[3],        # α/θ, β/α, γ/β                   │
│    asymmetry[1],          # AF7-AF8 alpha asymmetry          │
│    coherence[6],          # pairwise channel coherence       │
│    heart_rate[1],         # BPM from PPG                     │
│    hrv[1],                # heart rate variability            │
│    motion[3],             # accelerometer magnitude           │
│    signal_quality[4],     # per-channel quality               │
│    temporal_delta[5],     # change in bands vs 10s ago        │
│  ]  → 29-dimensional state vector                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              RL AGENT                                         │
│                                                              │
│  Policy: π(s_t) → a_t                                        │
│                                                              │
│  Actions (what the agent can do):                            │
│    ├─ Adjust binaural beat frequency                         │
│    ├─ Change ambient soundscape                              │
│    ├─ Trigger guided breathing prompt                        │
│    ├─ Adjust screen brightness/color temp                    │
│    ├─ Send notification / vibration                          │
│    ├─ Modify task difficulty (if in a game/study app)        │
│    └─ Log event + recommend break                            │
│                                                              │
│  Reward signals:                                             │
│    ├─ Sustained alpha increase → +r (relaxation goal)        │
│    ├─ Sustained beta/gamma → +r (focus goal)                 │
│    ├─ Reduced frontal asymmetry → +r (emotional balance)     │
│    ├─ Lower HRV entropy → +r (calm state)                    │
│    ├─ User explicit feedback (thumbs up/down) → +/-r         │
│    └─ Session duration without break → -r (fatigue penalty)  │
│                                                              │
│  Algorithm: PPO or SAC (continuous state, discrete actions)  │
│  Training: Online, per-user personalization                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              SELF-IMPROVING LOOP                              │
│                                                              │
│  1. Collect (s_t, a_t, r_t, s_{t+1}) tuples per session     │
│  2. Train policy on replay buffer after each session         │
│  3. Track long-term metrics:                                 │
│     - Average alpha power across sessions (improving?)       │
│     - Time to reach target state (getting faster?)           │
│     - Session engagement duration (longer?)                  │
│  4. Adapt reward weights based on user goals                 │
│  5. A/B test intervention strategies                         │
└─────────────────────────────────────────────────────────────┘
```

## Immediate Next Steps

1. ✅ Raw EEG streaming + display (DONE)
2. Feature extraction pipeline (band powers, ratios, asymmetry)
3. State vector computation at 1Hz
4. WebSocket API: stream state vectors to any consumer
5. Simple RL environment (OpenAI Gym interface)
6. First agent: "maximize user's alpha power" via audio interventions
