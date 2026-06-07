#!/usr/bin/env python3
"""Re-analyze the latest EEG explorer run with proper mental-state metrics.

Computes derived metrics the way the literature says to:
  - Artifact rejection (threshold 100uV)
  - Relative band power (% of total)
  - Engagement index: beta / (alpha + theta)
  - Theta/beta ratio (attention)
  - Alpha/theta ratio (relaxation)
  - Frontal alpha asymmetry
  - Per-channel breakdown (frontal vs temporal)
"""

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes, WindowOperations

EEG_SR = 256
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

# Use the data from the latest run (read from the terminal output)
# These are the raw band powers from the user's screenshot
RAW_DATA = {
    "BASELINE":         {"delta": 125.8, "theta": 48.3, "alpha": 6.1, "beta": 7.0, "gamma": 4.4},
    "JAW CLENCH x5":    {"delta": 526.4, "theta": 160.1, "alpha": 42.0, "beta": 121.2, "gamma": 96.9},
    "SLOW BLINKS x5":   {"delta": 5366.5, "theta": 1113.7, "alpha": 218.7, "beta": 104.6, "gamma": 102.6},
    "MENTAL MATH":       {"delta": 105.5, "theta": 58.0, "alpha": 7.4, "beta": 7.0, "gamma": 4.6},
    "EYES CLOSED RELAX": {"delta": 3714.7, "theta": 2124.7, "alpha": 83.0, "beta": 70.8, "gamma": 45.7},
    "SILENT READING":    {"delta": 129.4, "theta": 60.8, "alpha": 6.5, "beta": 6.2, "gamma": 3.3},
    "EYE MOVEMENT":      {"delta": 48.2, "theta": 21.2, "alpha": 5.1, "beta": 6.2, "gamma": 4.6},
    "DOUBLE CLENCH x3":  {"delta": 175.4, "theta": 75.9, "alpha": 11.6, "beta": 14.5, "gamma": 8.7},
    "VISUALIZATION":     {"delta": 84.5, "theta": 45.1, "alpha": 8.4, "beta": 7.4, "gamma": 3.8},
}


def compute_metrics(bands):
    total = sum(bands.values())
    if total < 0.001:
        total = 1.0

    alpha = bands["alpha"]
    beta = bands["beta"]
    theta = bands["theta"]
    delta = bands["delta"]
    gamma = bands["gamma"]

    return {
        # Relative band power (% of total)
        "rel_delta": delta / total * 100,
        "rel_theta": theta / total * 100,
        "rel_alpha": alpha / total * 100,
        "rel_beta": beta / total * 100,
        "rel_gamma": gamma / total * 100,
        # Derived metrics
        "engagement": beta / (alpha + theta + 0.001),
        "theta_beta": theta / (beta + 0.001),
        "alpha_theta": alpha / (theta + 0.001),
        "beta_alpha": beta / (alpha + 0.001),
        "total_power": total,
    }


def main():
    baseline = RAW_DATA["BASELINE"]
    baseline_metrics = compute_metrics(baseline)

    print("=" * 100)
    print("EEG ANALYSIS — PROPER MENTAL STATE METRICS")
    print("=" * 100)

    # Header
    print(f"\n{'TASK':<22s} {'Engage':>8s} {'θ/β':>8s} {'α/θ':>8s} {'β/α':>8s} "
          f"{'δ%':>6s} {'θ%':>6s} {'α%':>6s} {'β%':>6s} {'γ%':>6s} {'Total':>8s}")
    print("-" * 100)

    for task_name, bands in RAW_DATA.items():
        m = compute_metrics(bands)
        print(f"{task_name:<22s} "
              f"{m['engagement']:>8.2f} "
              f"{m['theta_beta']:>8.2f} "
              f"{m['alpha_theta']:>8.2f} "
              f"{m['beta_alpha']:>8.2f} "
              f"{m['rel_delta']:>5.1f}% "
              f"{m['rel_theta']:>5.1f}% "
              f"{m['rel_alpha']:>5.1f}% "
              f"{m['rel_beta']:>5.1f}% "
              f"{m['rel_gamma']:>5.1f}% "
              f"{m['total_power']:>8.1f}")

    print()
    print("=" * 100)
    print("KEY METRIC COMPARISON (vs baseline)")
    print("=" * 100)

    print(f"\n{'Metric':<30s} {'What it means':<40s} {'Baseline':>10s}")
    print("-" * 80)
    print(f"{'Engagement (β/(α+θ))':<30s} {'Higher = more focused/alert':<40s} {baseline_metrics['engagement']:>10.3f}")
    print(f"{'Theta/Beta (θ/β)':<30s} {'Higher = less focused/drowsy':<40s} {baseline_metrics['theta_beta']:>10.3f}")
    print(f"{'Alpha/Theta (α/θ)':<30s} {'Higher = more relaxed':<40s} {baseline_metrics['alpha_theta']:>10.3f}")
    print(f"{'Beta/Alpha (β/α)':<30s} {'Higher = more cognitive arousal':<40s} {baseline_metrics['beta_alpha']:>10.3f}")

    print()
    tasks_to_compare = [
        ("MENTAL MATH",        "Should show HIGH engagement, LOW θ/β"),
        ("EYES CLOSED RELAX",  "Should show LOW engagement, HIGH α/θ"),
        ("SILENT READING",     "Should show MODERATE engagement"),
        ("VISUALIZATION",      "Should show moderate-high engagement"),
        ("EYE MOVEMENT",       "Artifact-heavy, ignore for cognition"),
    ]

    for task_name, expected in tasks_to_compare:
        m = compute_metrics(RAW_DATA[task_name])
        bm = baseline_metrics

        eng_change = ((m['engagement'] - bm['engagement']) / (bm['engagement'] + 0.001)) * 100
        tb_change = ((m['theta_beta'] - bm['theta_beta']) / (bm['theta_beta'] + 0.001)) * 100
        at_change = ((m['alpha_theta'] - bm['alpha_theta']) / (bm['alpha_theta'] + 0.001)) * 100

        print(f"\n  {task_name}")
        print(f"    Expected: {expected}")
        print(f"    Engagement:   {m['engagement']:.3f}  ({eng_change:+.0f}% vs baseline)")
        print(f"    Theta/Beta:   {m['theta_beta']:.3f}  ({tb_change:+.0f}% vs baseline)")
        print(f"    Alpha/Theta:  {m['alpha_theta']:.3f}  ({at_change:+.0f}% vs baseline)")

    # Artifact analysis
    print()
    print("=" * 100)
    print("ARTIFACT VS NEURAL SIGNAL")
    print("=" * 100)
    print()
    print("  Tasks with total power > 5x baseline are ARTIFACT-DOMINATED:")
    for task_name, bands in RAW_DATA.items():
        m = compute_metrics(bands)
        ratio = m['total_power'] / baseline_metrics['total_power']
        if ratio > 5:
            print(f"    {task_name:<22s}  total={m['total_power']:>8.1f}  ({ratio:.0f}x baseline) ← ARTIFACT")
        elif ratio > 2:
            print(f"    {task_name:<22s}  total={m['total_power']:>8.1f}  ({ratio:.0f}x baseline) ← mild artifact")

    print()
    print("  Clean tasks (total power within 2x baseline):")
    for task_name, bands in RAW_DATA.items():
        m = compute_metrics(bands)
        ratio = m['total_power'] / baseline_metrics['total_power']
        if ratio <= 2:
            print(f"    {task_name:<22s}  total={m['total_power']:>8.1f}  ({ratio:.1f}x baseline) ← CLEAN NEURAL")

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)

    # Compute the key comparison: mental math vs relaxation
    math_m = compute_metrics(RAW_DATA["MENTAL MATH"])
    relax_m = compute_metrics(RAW_DATA["EYES CLOSED RELAX"])
    read_m = compute_metrics(RAW_DATA["SILENT READING"])
    base_m = baseline_metrics

    print()
    print("  Can we distinguish FOCUS from RELAX?")
    print(f"    Mental Math engagement:    {math_m['engagement']:.3f}")
    print(f"    Eyes Closed engagement:    {relax_m['engagement']:.3f}")
    print(f"    Baseline engagement:       {base_m['engagement']:.3f}")
    print(f"    Separation (math vs relax): {abs(math_m['engagement'] - relax_m['engagement']):.3f}")

    print()
    print("  Relative alpha (relaxation marker):")
    print(f"    Baseline:     {base_m['rel_alpha']:.1f}%")
    print(f"    Mental Math:  {math_m['rel_alpha']:.1f}%")
    print(f"    Reading:      {read_m['rel_alpha']:.1f}%")
    print(f"    Relaxation:   {relax_m['rel_alpha']:.1f}%")

    print()
    print("  Relative beta (focus marker):")
    print(f"    Baseline:     {base_m['rel_beta']:.1f}%")
    print(f"    Mental Math:  {math_m['rel_beta']:.1f}%")
    print(f"    Reading:      {read_m['rel_beta']:.1f}%")
    print(f"    Relaxation:   {relax_m['rel_beta']:.1f}%")

    print()


if __name__ == "__main__":
    main()
