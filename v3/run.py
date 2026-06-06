#!/usr/bin/env python3
"""Axiom v3 — Two boxes. Look at one. Focus to click.

Pipeline (all pure math, no LLM, runs at ~30Hz):
  1. Muse S EEG → engagement index β/(α+θ)
  2. Eyetrax gaze → which box are you looking at
  3. Engagement crosses threshold while fixating → CLICK

Three phases:
  1. Gaze calibration (eyetrax 5x5 dense grid)
  2. EEG calibration (measure YOUR baseline vs focus levels)
  3. Live — look at a box, focus to select it

Usage:
  python3 run.py
"""

import cv2
import numpy as np
import os
import sys
import time
from collections import deque

# ── EEG imports (from backend, but self-contained) ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Prevent Python's resource_tracker from unlinking the bridge's shared memory
# when this process exits. Only the bridge should manage its own shared memory.
from multiprocessing import resource_tracker
_original_register = resource_tracker.register
def _patched_register(name, rtype):
    if rtype == "shared_memory" and name in ("/muse_eeg_ring", "/muse_eeg_meta"):
        return
    _original_register(name, rtype)
resource_tracker.register = _patched_register

from muse_bridge import EEGBridgeClient, is_bridge_running
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

# ── Gaze imports ─────────────────────────────────────────────
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from screeninfo import get_monitors

# ── Constants ────────────────────────────────────────────────
EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

# How long you must sustain focus on a box to trigger a click
DWELL_SECONDS = 1.5
# How long after a click before the system is ready again
COOLDOWN_SECONDS = 2.0
# Engagement history window for smoothing
ENG_HISTORY_SIZE = 15  # ~0.5s at 30Hz


def get_screen():
    try:
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return 1470, 956


def find_camera():
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        ret, _ = cap.read()
        cap.release()
        if ret:
            return idx
    print("ERROR: No camera found")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════
# EEG FEATURE EXTRACTION (pure numpy/brainflow, <1ms)
# ═════════════════════════════════════════════════════════════

def compute_band_powers(data_1ch):
    """Band powers for a single channel. Input: float64 array, len >= 256."""
    if len(data_1ch) < EEG_SR:
        return {n: 0.0 for n, _, _ in BANDS}
    out = data_1ch.copy()
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
    nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
    psd = DataFilter.get_psd_welch(out, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
    return {n: float(DataFilter.get_band_power(psd, lo, hi)) for n, lo, hi in BANDS}


def compute_engagement(eeg_4ch_window):
    """Compute engagement index from a (4, N) EEG window. Returns float."""
    if eeg_4ch_window.shape[1] < EEG_SR:
        return 0.0
    # Use frontal channels (AF7=1, AF8=2) for engagement — less muscle noise
    bp1 = compute_band_powers(eeg_4ch_window[1])
    bp2 = compute_band_powers(eeg_4ch_window[2])
    alpha = (bp1["alpha"] + bp2["alpha"]) / 2
    theta = (bp1["theta"] + bp2["theta"]) / 2
    beta = (bp1["beta"] + bp2["beta"]) / 2
    return beta / (alpha + theta + 0.001)


# ═════════════════════════════════════════════════════════════
# EEG CALIBRATION — measure personal baseline vs focus
# ═════════════════════════════════════════════════════════════

def calibrate_eeg(eeg, sw, sh):
    """Measure the user's baseline and focus engagement levels.
    Returns (baseline_mean, baseline_std, focus_mean, focus_std, threshold)."""

    cv2.namedWindow("EEG Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("EEG Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    phases = [
        ("RELAX", "Relax. Stare at the dot. Think about nothing.", 8, "baseline"),
        ("FOCUS", "Count backwards from 300 by 7. Go fast.", 8, "focus"),
        ("RELAX 2", "Relax again. Clear your mind. Breathe.", 8, "baseline"),
        ("FOCUS 2", "Multiply: 13 x 17, then 23 x 19, then 37 x 11.", 8, "focus"),
    ]

    calibration_data = {"baseline": [], "focus": []}

    for phase_name, instruction, duration, label in phases:
        # Wait screen
        while True:
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            canvas[:] = (20, 6, 6)
            cv2.putText(canvas, "EEG CALIBRATION", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
            cv2.putText(canvas, phase_name, (sw // 2 - 100, sh // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 200, 100) if label == "baseline" else (0, 100, 255), 2)
            cv2.putText(canvas, instruction, (sw // 2 - 280, sh // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 190), 1)
            cv2.putText(canvas, "Press SPACE when ready", (sw // 2 - 140, sh // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.imshow("EEG Calibration", canvas)
            eeg.pull()
            if cv2.waitKey(30) & 0xFF == 32:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow("EEG Calibration")
                return None

        # Record phase
        start = time.time()
        while True:
            eeg.pull()
            window = eeg.get_window(2.0)
            elapsed = time.time() - start

            if window.shape[1] >= EEG_SR:
                eng = compute_engagement(window)
                calibration_data[label].append(eng)

            remaining = max(0, duration - elapsed)
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            canvas[:] = (20, 6, 6)
            cv2.putText(canvas, phase_name, (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 100) if label == "baseline" else (0, 100, 255), 2)
            cv2.putText(canvas, f"{remaining:.0f}s", (sw // 2 - 30, sh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 200, 100) if label == "baseline" else (0, 100, 255), 3)
            cv2.putText(canvas, instruction, (sw // 2 - 280, sh // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)
            # Fixation dot
            cv2.circle(canvas, (sw // 2, sh // 2 - 60), 5, (100, 200, 100), -1)
            # REC indicator
            pulse = int(128 + 127 * np.sin(elapsed * 4))
            cv2.circle(canvas, (sw - 30, 30), 5, (0, 0, pulse), -1)

            cv2.imshow("EEG Calibration", canvas)
            if cv2.waitKey(30) & 0xFF == 27:
                cv2.destroyWindow("EEG Calibration")
                return None
            if elapsed >= duration:
                break

    cv2.destroyWindow("EEG Calibration")

    # Compute threshold
    b_vals = calibration_data["baseline"]
    f_vals = calibration_data["focus"]

    if not b_vals or not f_vals:
        print("  Calibration failed — not enough data")
        return None

    b_mean, b_std = np.mean(b_vals), np.std(b_vals)
    f_mean, f_std = np.mean(f_vals), np.std(f_vals)

    # Threshold = midpoint between baseline and focus, biased toward focus
    # to avoid false positives
    threshold = b_mean + 0.65 * (f_mean - b_mean)

    # If focus isn't measurably higher than baseline, use a fixed offset
    if f_mean <= b_mean + b_std * 0.5:
        print("  WARNING: Focus and baseline are very close.")
        print("  Using baseline + 1 std as threshold.")
        threshold = b_mean + b_std

    return {
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "focus_mean": f_mean,
        "focus_std": f_std,
        "threshold": threshold,
    }


# ═════════════════════════════════════════════════════════════
# MAIN LOOP — Two boxes, gaze + focus to click
# ═════════════════════════════════════════════════════════════

def run_live(eeg, gaze_estimator, gaze_smoother, cam_idx, eeg_cal, sw, sh):
    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Axiom v3", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Axiom v3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    threshold = eeg_cal["threshold"]
    baseline_mean = eeg_cal["baseline_mean"]
    focus_mean = eeg_cal["focus_mean"]

    # Box layout — two big boxes, well separated
    gap = sw // 6
    box_w = (sw - 3 * gap) // 2
    box_h = sh // 2
    box_y = (sh - box_h) // 2 + 20
    left_box = (gap, box_y, box_w, box_h)
    right_box = (2 * gap + box_w, box_y, box_w, box_h)
    boxes = [left_box, right_box]
    box_labels = ["LEFT", "RIGHT"]
    box_colors = [(255, 150, 50), (50, 200, 255)]  # BGR: orange, cyan

    # State
    gaze_x, gaze_y = sw // 2, sh // 2
    gazed_box = -1
    eng_history = deque(maxlen=ENG_HISTORY_SIZE)
    current_engagement = 0.0

    # Dwell tracking per box
    dwell_start = [-1.0, -1.0]  # when focus dwell started for each box
    dwell_progress = [0.0, 0.0]

    last_click_time = 0.0
    click_log = []
    score = [0, 0]

    # Engagement bar scaling
    eng_min = max(0, baseline_mean - eeg_cal["baseline_std"])
    eng_max = focus_mean + eeg_cal["focus_std"]

    print(f"\n  Threshold: {threshold:.3f}")
    print(f"  Baseline:  {baseline_mean:.3f} ± {eeg_cal['baseline_std']:.3f}")
    print(f"  Focus:     {focus_mean:.3f} ± {eeg_cal['focus_std']:.3f}")
    print(f"\n  Look at a box. Focus hard to select it. ESC to quit.\n")

    while True:
        # ── 1. Pull EEG ──────────────────────────────────────
        eeg.pull()
        window = eeg.get_window(2.0)
        if window.shape[1] >= EEG_SR:
            eng = compute_engagement(window)
            eng_history.append(eng)
            current_engagement = np.mean(eng_history)

        # ── 2. Gaze tracking ─────────────────────────────────
        ret, frame = cap.read()
        if ret:
            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                raw = gaze_estimator.predict(np.array([features]))[0]
                sx, sy = gaze_smoother.step(int(raw[0]), int(raw[1]))
                gaze_x, gaze_y = sx, sy

        # ── 3. Hit test — which box? ─────────────────────────
        pad = 40
        gazed_box = -1
        for i, (bx, by, bw, bh) in enumerate(boxes):
            if (bx - pad <= gaze_x <= bx + bw + pad and
                    by - pad <= gaze_y <= by + bh + pad):
                gazed_box = i
                break

        # ── 4. Focus dwell — sustained engagement on a box ───
        now = time.time()
        in_cooldown = (now - last_click_time) < COOLDOWN_SECONDS
        above_threshold = current_engagement >= threshold

        for i in range(2):
            if gazed_box == i and above_threshold and not in_cooldown:
                if dwell_start[i] < 0:
                    dwell_start[i] = now
                dwell_progress[i] = min(1.0, (now - dwell_start[i]) / DWELL_SECONDS)
            else:
                # Reset dwell if not looking or not focused
                dwell_start[i] = -1.0
                dwell_progress[i] = max(0, dwell_progress[i] - 0.05)  # fade out

            # ── 5. CLICK — dwell complete ────────────────────
            if dwell_progress[i] >= 1.0:
                score[i] += 1
                click_log.append((now, box_labels[i], current_engagement))
                print(f"  CLICK: {box_labels[i]} (engagement={current_engagement:.3f})")
                last_click_time = now
                dwell_start[i] = -1.0
                dwell_progress[i] = 0.0

        # ── 6. Draw ──────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (20, 6, 6)

        # Header
        cv2.putText(canvas, "AXIOM v3", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 208, 232), 2)
        cv2.putText(canvas, f"LEFT: {score[0]}  RIGHT: {score[1]}", (sw // 2 - 80, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 190), 1)
        cv2.putText(canvas, "ESC quit | R recalibrate", (sw - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 120), 1)

        # Boxes
        for i, (bx, by, bw, bh) in enumerate(boxes):
            color = box_colors[i]
            is_gazed = (gazed_box == i)

            # Fill
            overlay = canvas.copy()
            fill_alpha = 0.25 if is_gazed else 0.08
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

            # Border
            border_color = (255, 255, 255) if is_gazed else color
            border_thick = 3 if is_gazed else 1
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_color, border_thick)

            # Label
            label = box_labels[i]
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.putText(canvas, label, (bx + (bw - ts[0]) // 2, by + bh // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255) if is_gazed else tuple(int(c * 0.6) for c in color), 2)

            # Score
            cv2.putText(canvas, str(score[i]),
                        (bx + bw // 2 - 10, by + bh // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 140), 2)

            # Dwell progress ring
            if dwell_progress[i] > 0.01:
                center = (bx + bw // 2, by + bh // 2 - 30)
                radius = 40
                angle = int(360 * dwell_progress[i])
                # Background ring
                cv2.ellipse(canvas, center, (radius, radius), 0, 0, 360, (40, 40, 60), 3)
                # Progress arc
                ring_color = (0, 255, 136) if dwell_progress[i] > 0.8 else color
                cv2.ellipse(canvas, center, (radius, radius), -90, 0, angle, ring_color, 4)

            # Flash on recent click
            if click_log and click_log[-1][1] == box_labels[i]:
                flash_age = now - click_log[-1][0]
                if flash_age < 0.5:
                    flash_alpha = 0.3 * (1 - flash_age / 0.5)
                    overlay2 = canvas.copy()
                    cv2.rectangle(overlay2, (bx, by), (bx + bw, by + bh), (0, 255, 136), -1)
                    cv2.addWeighted(overlay2, flash_alpha, canvas, 1 - flash_alpha, 0, canvas)

        # Gaze dot
        cv2.circle(canvas, (gaze_x, gaze_y), 5, (0, 212, 255), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 7, (255, 255, 255), 1)

        # Engagement bar (bottom)
        bar_x, bar_y = 40, sh - 50
        bar_w, bar_h = sw - 80, 20
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 50), -1)

        # Fill based on engagement
        eng_norm = np.clip((current_engagement - eng_min) / (eng_max - eng_min + 0.001), 0, 1)
        fill_w = int(bar_w * eng_norm)
        eng_color = (0, 255, 136) if above_threshold else (0, 160, 255)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), eng_color, -1)

        # Threshold marker
        thresh_norm = np.clip((threshold - eng_min) / (eng_max - eng_min + 0.001), 0, 1)
        thresh_x = bar_x + int(bar_w * thresh_norm)
        cv2.line(canvas, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_h + 5), (255, 255, 255), 2)

        # Labels
        cv2.putText(canvas, f"Engagement: {current_engagement:.3f}", (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 190), 1)
        cv2.putText(canvas, f"Threshold: {threshold:.3f}", (thresh_x - 40, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        status = "COOLDOWN" if in_cooldown else "FOCUSED" if above_threshold else "relaxed"
        status_color = (100, 100, 140) if in_cooldown else (0, 255, 136) if above_threshold else (100, 100, 140)
        cv2.putText(canvas, status, (sw - 120, sh - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        cv2.imshow("Axiom v3", canvas)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            # Recalibrate gaze
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(gaze_estimator, rows=5, cols=5,
                                       order="serpentine", pulse_d=0.8, cd_d=0.8,
                                       camera_index=cam_idx)
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Axiom v3", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Axiom v3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Final score: LEFT={score[0]}  RIGHT={score[1]}")
    print(f"  Total clicks: {len(click_log)}")
    for t, label, eng in click_log:
        print(f"    {label} @ engagement={eng:.3f}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    sw, sh = get_screen()
    print(f"Screen: {sw}x{sh}")

    # ── Check Muse bridge ────────────────────────────────────
    if not is_bridge_running():
        print("ERROR: Start muse_bridge.py first")
        print("  cd ../backend && python3 muse_bridge.py")
        sys.exit(1)

    eeg = EEGBridgeClient()
    eeg.start()
    print("EEG connected")

    # Warm up
    for _ in range(40):
        eeg.pull()
        time.sleep(0.05)

    # ── Phase 1: Gaze calibration ────────────────────────────
    cam_idx = find_camera()
    gaze = GazeEstimator(model_name="tiny_mlp")
    print("\n=== GAZE CALIBRATION ===")
    print("  Keep window focused! Move head slightly between dots.")
    run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                pulse_d=0.8, cd_d=0.8, camera_index=cam_idx)
    smoother = KalmanEMASmoother(make_kalman(), ema_alpha=0.3)
    print("  Gaze calibration done.\n")

    # ── Phase 2: EEG calibration ─────────────────────────────
    print("=== EEG CALIBRATION ===")
    print("  We'll measure your baseline and focus levels.")
    eeg_cal = calibrate_eeg(eeg, sw, sh)
    if eeg_cal is None:
        print("  EEG calibration aborted.")
        eeg.stop()
        return

    print(f"\n  Baseline engagement: {eeg_cal['baseline_mean']:.3f} ± {eeg_cal['baseline_std']:.3f}")
    print(f"  Focus engagement:    {eeg_cal['focus_mean']:.3f} ± {eeg_cal['focus_std']:.3f}")
    print(f"  Click threshold:     {eeg_cal['threshold']:.3f}")

    # ── Phase 3: Live ────────────────────────────────────────
    print("\n=== LIVE MODE ===")
    run_live(eeg, gaze, smoother, cam_idx, eeg_cal, sw, sh)

    eeg.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
