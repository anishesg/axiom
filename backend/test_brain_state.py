#!/usr/bin/env python3
"""Test the Brain State Engine with synthetic data based on real Muse S recordings.

Real session stats (from our live captures):
  TP9: std=608, AF7: std=33, AF8: std=26, TP10: std=629
  Bands: delta=536, theta=710, alpha=1599, beta=7156, gamma=8722
"""

import numpy as np
import time
from brain_state import BrainStateEngine, BrainState

EEG_SR = 256
np.random.seed(42)


def generate_eeg(seconds: float, state: str = "normal") -> np.ndarray:
    """Generate realistic synthetic EEG based on real Muse S stats."""
    n = int(EEG_SR * seconds)
    t = np.arange(n) / EEG_SR
    eeg = np.zeros((4, n))

    # Base frequency components
    for ch in range(4):
        delta = 8 * np.sin(2 * np.pi * 2.5 * t + ch * 0.5)
        theta = 6 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        alpha = 12 * np.sin(2 * np.pi * 10 * t + ch * 0.7)
        beta = 4 * np.sin(2 * np.pi * 22 * t + ch * 0.2)
        gamma = 2 * np.sin(2 * np.pi * 40 * t + ch * 0.9)
        noise = np.random.randn(n) * 3

        if state == "focused":
            beta *= 3.0
            gamma *= 2.5
            alpha *= 0.5
        elif state == "relaxed":
            alpha *= 3.0
            beta *= 0.4
            gamma *= 0.3
        elif state == "drowsy":
            theta *= 3.0
            alpha *= 2.0
            beta *= 0.3
        elif state == "context_switch":
            # Alpha rises then theta spikes
            alpha *= (1 + 2 * np.exp(-((t - seconds / 2) ** 2) / 0.1))
            theta *= (1 + 3 * np.exp(-((t - seconds / 2 + 0.3) ** 2) / 0.1))
        elif state == "jaw_clench":
            # Massive EMG artifact in temporal channels
            clench_start = int(n * 0.7)
            clench_end = clench_start + 25
            if ch in (0, 3):  # TP9, TP10
                eeg[ch, clench_start:clench_end] = np.random.randn(clench_end - clench_start) * 800

        eeg[ch] += delta + theta + alpha + beta + gamma + noise

        # Temporal channels (TP9, TP10) are noisier per real data
        if ch in (0, 3):
            eeg[ch] *= 15
        else:
            eeg[ch] *= 1.0

    return eeg


def test_state_classification():
    engine = BrainStateEngine()
    print("=" * 60)
    print("AXIOM BRAIN STATE ENGINE — SIMULATION TEST")
    print("=" * 60)

    scenarios = [
        ("baseline",       "normal",         "Baseline resting state"),
        ("focused",        "focused",         "Deep focus (high beta/gamma)"),
        ("relaxed",        "relaxed",         "Eyes-closed relaxation (high alpha)"),
        ("drowsy",         "drowsy",          "Drowsy / low engagement (high theta)"),
        ("context_switch", "context_switch",  "Task switching (alpha→theta transition)"),
        ("jaw_clench",     "jaw_clench",      "Jaw clench (EMG artifact)"),
    ]

    for name, synth_state, description in scenarios:
        eeg = generate_eeg(2.0, state=synth_state)
        state = engine.process(eeg, timestamp=time.time())

        print(f"\n{'─' * 60}")
        print(f"  SCENARIO: {description}")
        print(f"{'─' * 60}")
        print(f"  engagement:    {state.engagement:.3f}  {'█' * int(state.engagement * 20)}")
        print(f"  focus:         {state.focus:.3f}  {'█' * int(state.focus * 20)}")
        print(f"  relaxation:    {state.relaxation:.3f}  {'█' * int(state.relaxation * 20)}")
        print(f"  cognitive_load:{state.cognitive_load:.3f}  {'█' * int(state.cognitive_load * 20)}")
        print(f"  valence:       {state.valence:.3f}  {'<neg' if state.valence < 0.4 else (' neu' if state.valence < 0.6 else ' pos>')}")
        print(f"  jaw_clench:    {state.jaw_clench}")
        print(f"  double_clench: {state.double_clench}")
        print(f"  context_switch:{state.context_switch:.3f}")
        print(f"  error_response:{state.error_response:.3f}")
        print(f"  quality:       {['%.2f' % q for q in state.signal_quality]}")


def test_temporal_tracking():
    """Test that the engine tracks state changes over time."""
    engine = BrainStateEngine()
    print(f"\n\n{'=' * 60}")
    print("TEMPORAL STATE TRACKING — 10s scenario")
    print("=" * 60)
    print("  0-3s: relaxed → 3-6s: focused → 6-8s: clench → 8-10s: relaxed")
    print()

    timeline = [
        (3.0, "relaxed"),
        (3.0, "focused"),
        (2.0, "jaw_clench"),
        (2.0, "relaxed"),
    ]

    t = 0.0
    for duration, synth_state in timeline:
        # Process in 0.5s windows (simulating 2Hz state updates)
        steps = int(duration / 0.5)
        for _ in range(steps):
            eeg = generate_eeg(1.0, state=synth_state)  # 1s window
            state = engine.process(eeg, timestamp=t)
            clench_str = " ** CLENCH **" if state.jaw_clench else ""
            print(f"  t={t:5.1f}s | eng={state.engagement:.2f} foc={state.focus:.2f} "
                  f"rel={state.relaxation:.2f} load={state.cognitive_load:.2f} "
                  f"val={state.valence:.2f}{clench_str}")
            t += 0.5

    print("\nDone.")


if __name__ == "__main__":
    test_state_classification()
    test_temporal_tracking()
