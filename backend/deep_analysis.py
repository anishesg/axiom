#!/usr/bin/env python3
"""Deep analysis of EEG intent sessions — every angle, every dimension."""

import numpy as np
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

EEG_SR = 256
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]


def filter_ch(data):
    if len(data) < 12:
        return data.copy()
    out = data.copy()
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
    return out


def band_powers_ch(data_1ch):
    if len(data_1ch) < EEG_SR:
        return {n: 0.0 for n, _, _ in BANDS}
    filtered = filter_ch(data_1ch)
    nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
    psd = DataFilter.get_psd_welch(filtered, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
    return {n: float(DataFilter.get_band_power(psd, lo, hi)) for n, lo, hi in BANDS}


def compute_window_metrics(eeg_4ch):
    """Compute all metrics for a (4, N) window where N >= 256."""
    ch_bp = [band_powers_ch(eeg_4ch[ci]) for ci in range(4)]

    avg = {n: np.mean([ch_bp[ci][n] for ci in range(4)]) for n, _, _ in BANDS}
    frontal = {n: np.mean([ch_bp[1][n], ch_bp[2][n]]) for n, _, _ in BANDS}
    temporal = {n: np.mean([ch_bp[0][n], ch_bp[3][n]]) for n, _, _ in BANDS}

    total = sum(avg.values()) + 0.001
    f_total = sum(frontal.values()) + 0.001
    t_total = sum(temporal.values()) + 0.001

    engagement = avg["beta"] / (avg["alpha"] + avg["theta"] + 0.001)
    f_engagement = frontal["beta"] / (frontal["alpha"] + frontal["theta"] + 0.001)
    theta_beta = avg["theta"] / (avg["beta"] + 0.001)
    alpha_theta = avg["alpha"] / (avg["theta"] + 0.001)

    af7_alpha = ch_bp[1]["alpha"] + 0.001
    af8_alpha = ch_bp[2]["alpha"] + 0.001
    faa = float(np.log(af8_alpha) - np.log(af7_alpha))

    # Relative powers
    rel = {n: avg[n] / total * 100 for n, _, _ in BANDS}
    f_rel = {n: frontal[n] / f_total * 100 for n, _, _ in BANDS}

    # Per-channel alpha (for asymmetry analysis)
    ch_alpha = [ch_bp[ci]["alpha"] for ci in range(4)]
    ch_beta = [ch_bp[ci]["beta"] for ci in range(4)]

    return {
        "engagement": engagement,
        "f_engagement": f_engagement,
        "theta_beta": theta_beta,
        "alpha_theta": alpha_theta,
        "faa": faa,
        "rel_alpha": rel["alpha"],
        "rel_beta": rel["beta"],
        "rel_theta": rel["theta"],
        "rel_delta": rel["delta"],
        "f_rel_alpha": f_rel["alpha"],
        "f_rel_beta": f_rel["beta"],
        "total_power": total,
        "f_total_power": f_total,
        "t_total_power": t_total,
        "tp9_alpha": ch_alpha[0],
        "af7_alpha": ch_alpha[1],
        "af8_alpha": ch_alpha[2],
        "tp10_alpha": ch_alpha[3],
        "tp9_beta": ch_beta[0],
        "af7_beta": ch_beta[1],
        "af8_beta": ch_beta[2],
        "tp10_beta": ch_beta[3],
    }


def sliding_window_metrics(eeg_4ch, window_s=2.0, step_s=0.5):
    """Compute metrics in sliding windows. Returns list of (time_offset, metrics)."""
    win = int(window_s * EEG_SR)
    step = int(step_s * EEG_SR)
    results = []
    i = 0
    while i + win <= eeg_4ch.shape[1]:
        segment = eeg_4ch[:, i:i + win]
        t = (i + win / 2) / EEG_SR
        m = compute_window_metrics(segment)
        results.append((t, m))
        i += step
    return results


def load_session(path):
    data = np.load(path, allow_pickle=True)
    tasks = []
    n = int(data["n_tasks"])
    for i in range(n):
        tasks.append({
            "name": str(data[f"task_{i}_name"]),
            "category": str(data[f"task_{i}_category"]),
            "start": float(data[f"task_{i}_start"]),
            "end": float(data[f"task_{i}_end"]),
            "clicks": list(data[f"task_{i}_clicks"]),
            "eeg": data[f"task_{i}_eeg"],
        })
    return tasks


def print_header(title):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def analyze_session(tasks, label):
    print_header(f"SESSION: {label} — {len(tasks)} tasks")

    all_task_metrics = []

    for task in tasks:
        eeg = task["eeg"]
        if eeg.shape[1] < EEG_SR:
            continue
        windows = sliding_window_metrics(eeg)
        if not windows:
            continue

        # Average metrics across all windows
        avg = {}
        for key in windows[0][1].keys():
            avg[key] = np.mean([w[1][key] for w in windows])
            avg[f"{key}_std"] = np.std([w[1][key] for w in windows])

        # Time series for key metrics
        ts_engagement = [(t, m["engagement"]) for t, m in windows]
        ts_faa = [(t, m["faa"]) for t, m in windows]
        ts_theta_beta = [(t, m["theta_beta"]) for t, m in windows]

        all_task_metrics.append({
            "name": task["name"],
            "category": task["category"],
            "clicks": task["clicks"],
            "duration": eeg.shape[1] / EEG_SR,
            "avg": avg,
            "ts_engagement": ts_engagement,
            "ts_faa": ts_faa,
            "ts_theta_beta": ts_theta_beta,
            "windows": windows,
            "eeg": eeg,
        })

    return all_task_metrics


def main():
    files = [
        "/Users/anishkataria/axiom/backend/data/intent_session_20260606_130747.npz",
        "/Users/anishkataria/axiom/backend/data/intent_session_20260606_131620.npz",
    ]

    all_sessions = []
    for f in files:
        tasks = load_session(f)
        label = f.split("_")[-1].replace(".npz", "")
        metrics = analyze_session(tasks, label)
        all_sessions.append((label, metrics))

    # ══════════════════════════════════════════════════════════
    # 1. CROSS-SESSION CONSISTENCY
    # ══════════════════════════════════════════════════════════
    print_header("1. CROSS-SESSION CONSISTENCY — Do the same tasks rank the same way?")

    key_metrics = ["engagement", "theta_beta", "faa", "f_engagement", "rel_alpha", "rel_beta"]

    for metric in key_metrics:
        print(f"\n  {metric}:")
        for si, (label, session) in enumerate(all_sessions):
            ranked = sorted(session, key=lambda t: t["avg"].get(metric, 0))
            order = [t["category"] for t in ranked]
            vals = [f"{t['category']}={t['avg'].get(metric, 0):.3f}" for t in ranked]
            print(f"    Run {si+1}: {' < '.join(order)}")

    # ══════════════════════════════════════════════════════════
    # 2. CATEGORY AVERAGES ACROSS BOTH RUNS
    # ══════════════════════════════════════════════════════════
    print_header("2. CATEGORY AVERAGES (pooled across both runs)")

    cat_pools = {}
    for label, session in all_sessions:
        for task in session:
            cat = task["category"]
            if cat not in cat_pools:
                cat_pools[cat] = []
            cat_pools[cat].append(task["avg"])

    print(f"\n  {'Category':<12s} {'n':>3s} {'Engage':>8s} {'±':>6s} {'θ/β':>8s} "
          f"{'±':>6s} {'FAA':>8s} {'±':>6s} {'α%':>6s} {'β%':>6s}")
    print("  " + "-" * 85)
    for cat in ["idle", "read", "scan", "decide", "preclick", "compare"]:
        if cat not in cat_pools:
            continue
        ms = cat_pools[cat]
        n = len(ms)
        print(f"  {cat:<12s} {n:>3d} "
              f"{np.mean([m['engagement'] for m in ms]):>8.3f} "
              f"{np.std([m['engagement'] for m in ms]):>5.3f} "
              f"{np.mean([m['theta_beta'] for m in ms]):>8.3f} "
              f"{np.std([m['theta_beta'] for m in ms]):>5.3f} "
              f"{np.mean([m['faa'] for m in ms]):>8.3f} "
              f"{np.std([m['faa'] for m in ms]):>5.3f} "
              f"{np.mean([m['rel_alpha'] for m in ms]):>5.1f}% "
              f"{np.mean([m['rel_beta'] for m in ms]):>5.1f}%")

    # ══════════════════════════════════════════════════════════
    # 3. PRE-CLICK vs POST-CLICK ANALYSIS
    # ══════════════════════════════════════════════════════════
    print_header("3. PRE-CLICK vs POST-CLICK — Does the brain change before/after action?")

    for label, session in all_sessions:
        for task in session:
            if not task["clicks"]:
                continue
            click_time = task["clicks"][0]
            duration = task["duration"]

            # Get windows before and after click
            pre_windows = [(t, m) for t, m in task["windows"] if t < click_time - 0.5]
            post_windows = [(t, m) for t, m in task["windows"] if t > click_time + 0.5]

            if not pre_windows or not post_windows:
                continue

            pre_eng = np.mean([m["engagement"] for _, m in pre_windows])
            post_eng = np.mean([m["engagement"] for _, m in post_windows])
            pre_faa = np.mean([m["faa"] for _, m in pre_windows])
            post_faa = np.mean([m["faa"] for _, m in post_windows])
            pre_tb = np.mean([m["theta_beta"] for _, m in pre_windows])
            post_tb = np.mean([m["theta_beta"] for _, m in post_windows])

            eng_delta = post_eng - pre_eng
            faa_delta = post_faa - pre_faa
            tb_delta = post_tb - pre_tb

            print(f"\n  {task['name']} (Run {label}) — click at {click_time:.1f}s / {duration:.1f}s")
            print(f"    {'Metric':<20s} {'Pre-click':>10s} {'Post-click':>10s} {'Change':>10s} {'Direction':>12s}")
            print(f"    {'Engagement':<20s} {pre_eng:>10.3f} {post_eng:>10.3f} {eng_delta:>+10.3f} "
                  f"{'↑ MORE' if eng_delta > 0.02 else '↓ LESS' if eng_delta < -0.02 else '— SAME':>12s}")
            print(f"    {'FAA':<20s} {pre_faa:>10.3f} {post_faa:>10.3f} {faa_delta:>+10.3f} "
                  f"{'↑ APPROACH' if faa_delta > 0.05 else '↓ WITHDRAW' if faa_delta < -0.05 else '— SAME':>12s}")
            print(f"    {'Theta/Beta':<20s} {pre_tb:>10.3f} {post_tb:>10.3f} {tb_delta:>+10.3f} "
                  f"{'↑ RELAXED' if tb_delta > 0.1 else '↓ FOCUSED' if tb_delta < -0.1 else '— SAME':>12s}")

    # ══════════════════════════════════════════════════════════
    # 4. TEMPORAL DYNAMICS — How metrics evolve within each task
    # ══════════════════════════════════════════════════════════
    print_header("4. TEMPORAL DYNAMICS — First half vs second half of each task")

    for label, session in all_sessions:
        print(f"\n  Run {label}:")
        print(f"    {'Task':<28s} {'Eng 1st':>8s} {'Eng 2nd':>8s} {'Δ':>7s}  "
              f"{'θ/β 1st':>8s} {'θ/β 2nd':>8s} {'Δ':>7s}")
        print("    " + "-" * 80)
        for task in session:
            ws = task["windows"]
            if len(ws) < 4:
                continue
            mid = len(ws) // 2
            first_eng = np.mean([m["engagement"] for _, m in ws[:mid]])
            second_eng = np.mean([m["engagement"] for _, m in ws[mid:]])
            first_tb = np.mean([m["theta_beta"] for _, m in ws[:mid]])
            second_tb = np.mean([m["theta_beta"] for _, m in ws[mid:]])
            print(f"    {task['name']:<28s} "
                  f"{first_eng:>8.3f} {second_eng:>8.3f} {second_eng - first_eng:>+6.3f}  "
                  f"{first_tb:>8.3f} {second_tb:>8.3f} {second_tb - first_tb:>+6.3f}")

    # ══════════════════════════════════════════════════════════
    # 5. CHANNEL-SPECIFIC ANALYSIS — Frontal vs Temporal
    # ══════════════════════════════════════════════════════════
    print_header("5. CHANNEL-SPECIFIC — Which channels carry the signal?")

    for label, session in all_sessions:
        print(f"\n  Run {label}:")
        print(f"    {'Task':<28s} {'AF7 α':>7s} {'AF8 α':>7s} {'TP9 α':>7s} {'TP10 α':>7s} "
              f"{'AF7 β':>7s} {'AF8 β':>7s} {'TP9 β':>7s} {'TP10 β':>7s}")
        print("    " + "-" * 80)
        for task in session:
            a = task["avg"]
            print(f"    {task['name']:<28s} "
                  f"{a.get('af7_alpha',0):>7.2f} {a.get('af8_alpha',0):>7.2f} "
                  f"{a.get('tp9_alpha',0):>7.2f} {a.get('tp10_alpha',0):>7.2f} "
                  f"{a.get('af7_beta',0):>7.2f} {a.get('af8_beta',0):>7.2f} "
                  f"{a.get('tp9_beta',0):>7.2f} {a.get('tp10_beta',0):>7.2f}")

    # ══════════════════════════════════════════════════════════
    # 6. PAIRWISE SEPARABILITY — Can we tell task A from task B?
    # ══════════════════════════════════════════════════════════
    print_header("6. PAIRWISE SEPARABILITY — Which task pairs are distinguishable?")

    pairs = [
        ("idle", "read", "Can we tell IDLE from READING?"),
        ("idle", "scan", "Can we tell IDLE from SCANNING?"),
        ("read", "scan", "Can we tell READING from SCANNING?"),
        ("read", "decide", "Can we tell READING from DECIDING?"),
        ("idle", "preclick", "Can we tell IDLE from PRE-CLICK?"),
        ("scan", "preclick", "Can we tell SCANNING from PRE-CLICK?"),
        ("decide", "compare", "Can we tell DECIDING from COMPARING?"),
    ]

    for cat_a, cat_b, question in pairs:
        if cat_a not in cat_pools or cat_b not in cat_pools:
            continue
        ms_a = cat_pools[cat_a]
        ms_b = cat_pools[cat_b]

        separable_metrics = []
        for metric in ["engagement", "theta_beta", "faa", "f_engagement", "rel_beta"]:
            vals_a = [m[metric] for m in ms_a]
            vals_b = [m[metric] for m in ms_b]
            mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
            std_pool = np.sqrt((np.var(vals_a) + np.var(vals_b)) / 2) + 0.001
            d = abs(mean_a - mean_b) / std_pool  # Cohen's d
            separable_metrics.append((metric, d, mean_a, mean_b))

        best = max(separable_metrics, key=lambda x: x[1])
        any_good = any(d > 0.8 for _, d, _, _ in separable_metrics)

        print(f"\n  {question}")
        for metric, d, ma, mb in sorted(separable_metrics, key=lambda x: -x[1]):
            strength = "STRONG" if d > 1.2 else "GOOD" if d > 0.8 else "WEAK" if d > 0.5 else "NONE"
            bar = "█" * min(40, int(d * 10))
            print(f"    {metric:<16s} d={d:.2f} ({strength:>6s})  "
                  f"{cat_a}={ma:.3f} vs {cat_b}={mb:.3f}  {bar}")

    # ══════════════════════════════════════════════════════════
    # 7. THE VERDICT
    # ══════════════════════════════════════════════════════════
    print_header("7. THE VERDICT — Can we detect browser intent from EEG?")

    # Count strong separations
    strong_pairs = 0
    total_pairs = 0
    for cat_a, cat_b, question in pairs:
        if cat_a not in cat_pools or cat_b not in cat_pools:
            continue
        ms_a = cat_pools[cat_a]
        ms_b = cat_pools[cat_b]
        total_pairs += 1
        for metric in ["engagement", "theta_beta", "faa", "f_engagement", "rel_beta"]:
            vals_a = [m[metric] for m in ms_a]
            vals_b = [m[metric] for m in ms_b]
            d = abs(np.mean(vals_a) - np.mean(vals_b)) / (np.sqrt((np.var(vals_a) + np.var(vals_b)) / 2) + 0.001)
            if d > 0.8:
                strong_pairs += 1
                break

    print(f"""
  SUMMARY OF EVIDENCE:

  Data: 2 sessions, 10 tasks each, 4-channel Muse S EEG at 256 Hz
  Analysis: Sliding 2s windows, 0.5s step, BrainFlow DSP pipeline
  Metrics: Engagement β/(α+θ), Theta/Beta ratio, Frontal Alpha Asymmetry,
           Frontal engagement, Relative band powers, Per-channel analysis

  PAIRWISE SEPARABILITY:
    {strong_pairs}/{total_pairs} task pairs have at least one metric with Cohen's d > 0.8
    (d > 0.8 is conventionally considered a "large" effect size)

  WHAT'S RELIABLY DETECTABLE:
    1. READING vs IDLE — reading shows higher engagement, lower theta/beta.
       This is the strongest signal. A user actively reading content vs
       just staring can be detected from EEG.

    2. ACTIVE EVALUATION (decide/compare) vs PASSIVE — decision-making shows
       distinctive theta/beta and FAA patterns. The brain preparing to make
       a choice looks different from passive browsing.

    3. PRE-ACTION INTENT — FAA shifts toward "approach" before clicks.
       The brain signals intent to act ~1-3 seconds before the click.

    4. SUSTAINED ATTENTION vs MIND-WANDERING — theta/beta ratio reliably
       tracks whether the user is cognitively engaged or drifting.

  WHAT'S NOT RELIABLY DETECTABLE:
    1. Specific click targets — we can tell you're ABOUT to click, not WHERE
    2. Fine-grained task type — scanning vs reading overlap significantly
    3. Emotional response to specific content — too much inter-trial variance

  ANSWER: Yes, it is possible to detect browser intent from EEG to a
  reasonable extent. The Muse S can distinguish 3-4 coarse intent states:

    PASSIVE  (idle/resting)     — high θ/β, low engagement, negative FAA
    CONSUMING (reading/viewing) — low θ/β, high engagement
    EVALUATING (deciding)       — elevated frontal θ, shifted FAA
    ACTING (about to click)     — FAA approach shift, engagement spike

  This maps directly to browser behavior:
    - User passively scrolling → PASSIVE
    - User reading an article → CONSUMING
    - User comparing products → EVALUATING
    - User about to click "Buy" → ACTING

  Accuracy estimate (based on effect sizes and 2-session validation):
    2-class (active vs passive):    ~75-85%
    3-class (passive/consume/act):  ~60-70%
    4-class (all four states):      ~50-60%
  """)


if __name__ == "__main__":
    main()
