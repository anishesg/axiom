#!/usr/bin/env python3
"""Gaze-Select: Intelligent element selection from coarse gaze.

The key insight: we don't need pixel-perfect gaze accuracy.
We need to correctly identify WHICH ELEMENT the user is looking at.

The page is divided into element zones. Coarse gaze + attention signals
(dwell time, directional stability, engagement) produce a per-element
confidence score. When confidence crosses threshold → that element
is selected.

This is a standalone calibration + testing environment.
Once it feels right, it integrates back into Axiom.

Usage:
    python3 run.py              # with Muse EEG
    python3 run.py --no-eeg     # gaze only, no Muse needed
"""

import cv2
import math
import numpy as np
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

# ── Gaze ─────────────────────────────────────────────────────
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from screeninfo import get_monitors

# ── EEG (optional) ───────────────────────────────────────────
USE_EEG = "--no-eeg" not in sys.argv
if USE_EEG:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from multiprocessing import resource_tracker
    _orig = resource_tracker.register
    def _patched(name, rtype):
        if rtype == "shared_memory" and name in ("/muse_eeg_ring", "/muse_eeg_meta"):
            return
        _orig(name, rtype)
    resource_tracker.register = _patched
    from muse_bridge import EEGBridgeClient, is_bridge_running
    from brainflow.data_filter import (
        DataFilter, FilterTypes, DetrendOperations,
        NoiseTypes, WindowOperations,
    )


# ═════════════════════════════════════════════════════════════
# ONE EURO FILTER — best-in-class gaze smoothing
# ═════════════════════════════════════════════════════════════

class OneEuroFilter:
    """Speed-adaptive low-pass filter. Smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Derivative
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ═════════════════════════════════════════════════════════════
# ELEMENT ZONE — represents a selectable region on screen
# ═════════════════════════════════════════════════════════════

@dataclass
class Element:
    id: int
    label: str
    x: int
    y: int
    w: int
    h: int
    color: tuple  # BGR

    # Live state (updated each frame)
    confidence: float = 0.0
    dwell_time: float = 0.0
    gaze_in: bool = False
    highlighted: bool = False
    selected: bool = False
    last_enter: float = 0.0
    stability: float = 0.0  # how stable gaze is within this element

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2


# ═════════════════════════════════════════════════════════════
# ATTENTION ENGINE — computes per-element confidence
# ═════════════════════════════════════════════════════════════

class AttentionEngine:
    """Combines gaze position, dwell time, stability, and EEG engagement
    into a per-element confidence score."""

    def __init__(self):
        # Tunable parameters
        self.dwell_weight = 0.4       # how much dwell time matters
        self.stability_weight = 0.3   # how much gaze stability matters
        self.proximity_weight = 0.2   # how much distance-to-center matters
        self.eeg_weight = 0.1         # how much EEG engagement matters

        # Thresholds
        self.highlight_threshold = 0.3    # confidence to start subtle highlight
        self.select_threshold = 0.75      # confidence to select (fill ring)
        self.switch_threshold = 0.15      # minimum confidence to switch away from current

        # Timing
        self.dwell_saturate = 2.0    # seconds of dwell to reach max score
        self.decay_rate = 0.92       # per-frame decay when not gazed

        # Gaze velocity tracking (for stability)
        self.gaze_history = deque(maxlen=15)

        # Currently highlighted element
        self.active_id = -1

    def update(self, elements: list[Element], gaze_x: float, gaze_y: float,
               engagement: float, dt: float):
        """Update confidence for all elements. Returns id of selected element or -1."""

        now = time.time()
        self.gaze_history.append((gaze_x, gaze_y, now))

        # Compute gaze velocity (stability metric)
        gaze_speed = 0.0
        if len(self.gaze_history) >= 3:
            pts = list(self.gaze_history)
            dists = [math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                     for i in range(1, len(pts))]
            time_span = pts[-1][2] - pts[0][2]
            if time_span > 0:
                gaze_speed = sum(dists) / time_span  # px/sec

        # Stability: inverse of speed, normalized to [0, 1]
        # < 50 px/s = very stable (1.0), > 500 px/s = saccade (0.0)
        stability = max(0, min(1, 1.0 - (gaze_speed - 50) / 450))

        selected_id = -1

        for el in elements:
            # Distance from gaze to element center
            dist = math.hypot(gaze_x - el.cx, gaze_y - el.cy)

            # Is gaze inside the element (with padding)?
            pad = 40
            in_element = (el.x - pad <= gaze_x <= el.x + el.w + pad and
                          el.y - pad <= gaze_y <= el.y + el.h + pad)

            # Proximity score: 1.0 at center, fading toward edges
            max_dist = math.hypot(el.w, el.h) / 2 + pad
            proximity = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0

            if in_element:
                if not el.gaze_in:
                    el.last_enter = now
                    el.gaze_in = True
                el.dwell_time = now - el.last_enter
                el.stability = stability
            else:
                el.gaze_in = False
                el.dwell_time = 0.0
                el.stability = 0.0

            # Compute confidence components
            dwell_score = min(1.0, el.dwell_time / self.dwell_saturate)
            stability_score = el.stability
            proximity_score = proximity
            eeg_score = max(0, min(1, engagement))

            if el.gaze_in:
                # Build up confidence
                raw = (self.dwell_weight * dwell_score +
                       self.stability_weight * stability_score +
                       self.proximity_weight * proximity_score +
                       self.eeg_weight * eeg_score)
                # Smooth transition: blend toward raw score
                el.confidence = el.confidence * 0.7 + raw * 0.3
            else:
                # Decay confidence
                el.confidence *= self.decay_rate

            # Clamp
            el.confidence = max(0, min(1, el.confidence))

            # Highlight state
            el.highlighted = el.confidence >= self.highlight_threshold

            # Selection check
            if el.confidence >= self.select_threshold:
                selected_id = el.id
                el.selected = True
            else:
                el.selected = False

        return selected_id


# ═════════════════════════════════════════════════════════════
# EEG HELPERS
# ═════════════════════════════════════════════════════════════

EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

def compute_engagement(eeg_window):
    if not USE_EEG or eeg_window is None or eeg_window.shape[1] < EEG_SR:
        return 0.5  # neutral

    def bp(data):
        if len(data) < EEG_SR:
            return {n: 0.0 for n, _, _ in BANDS}
        out = data.copy()
        DataFilter.detrend(out, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4,
                                    FilterTypes.BUTTERWORTH.value, 0.0)
        DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
        nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
        psd = DataFilter.get_psd_welch(out, nfft, nfft // 2, EEG_SR,
                                        WindowOperations.HANNING.value)
        return {n: float(DataFilter.get_band_power(psd, lo, hi)) for n, lo, hi in BANDS}

    # Frontal channels
    bp1 = bp(eeg_window[1])
    bp2 = bp(eeg_window[2])
    alpha = (bp1["alpha"] + bp2["alpha"]) / 2
    theta = (bp1["theta"] + bp2["theta"]) / 2
    beta = (bp1["beta"] + bp2["beta"]) / 2
    return beta / (alpha + theta + 0.001)


# ═════════════════════════════════════════════════════════════
# RENDERING
# ═════════════════════════════════════════════════════════════

def draw_element(canvas, el: Element, now: float):
    """Draw an element with confidence-based progressive highlighting."""
    c = el.confidence

    # Background fill — intensity based on confidence
    if c > 0.01:
        overlay = canvas.copy()
        fill_alpha = 0.05 + c * 0.25
        cv2.rectangle(overlay, (el.x, el.y), (el.x + el.w, el.y + el.h),
                      el.color, -1)
        cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

    # Border
    if c >= 0.75:
        # Strong — white glow
        cv2.rectangle(canvas, (el.x, el.y), (el.x + el.w, el.y + el.h),
                      (255, 255, 255), 3)
    elif c >= 0.3:
        # Medium — colored
        cv2.rectangle(canvas, (el.x, el.y), (el.x + el.w, el.y + el.h),
                      el.color, 2)
    else:
        # Faint
        dim = tuple(int(v * 0.3) for v in el.color)
        cv2.rectangle(canvas, (el.x, el.y), (el.x + el.w, el.y + el.h),
                      dim, 1)

    # Label
    label_alpha = 0.3 + c * 0.7
    label_color = tuple(int(v * label_alpha) for v in (255, 255, 255))
    ts = cv2.getTextSize(el.label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(canvas, el.label,
                (el.cx - ts[0] // 2, el.cy + ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # Confidence ring (when building toward selection)
    if c >= 0.3:
        radius = 30
        angle = int(360 * min(1, c / 0.75))
        ring_color = (0, 255, 136) if c >= 0.6 else el.color
        cv2.ellipse(canvas, (el.cx, el.cy - 40), (radius, radius),
                    -90, 0, 360, (40, 40, 60), 2)
        cv2.ellipse(canvas, (el.cx, el.cy - 40), (radius, radius),
                    -90, 0, angle, ring_color, 3)

    # Selection flash
    if el.selected:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (el.x, el.y), (el.x + el.w, el.y + el.h),
                      (0, 255, 136), -1)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    # Confidence label (small)
    cv2.putText(canvas, f"{c:.0%}",
                (el.x + 8, el.y + el.h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 120), 1)


# ═════════════════════════════════════════════════════════════
# LAYOUTS — different element arrangements to test
# ═════════════════════════════════════════════════════════════

def layout_two_boxes(sw, sh):
    """Two big boxes — easiest test."""
    gap = sw // 5
    bw = (sw - 3 * gap) // 2
    bh = sh // 2
    by = (sh - bh) // 2 + 20
    return [
        Element(0, "LEFT", gap, by, bw, bh, (255, 150, 50)),
        Element(1, "RIGHT", 2 * gap + bw, by, bw, bh, (50, 200, 255)),
    ]

def layout_four_quadrants(sw, sh):
    """Four quadrants — medium difficulty."""
    margin = 60
    gap = 20
    bw = (sw - 2 * margin - gap) // 2
    bh = (sh - 120 - gap) // 2
    top = 70
    return [
        Element(0, "TOP-LEFT", margin, top, bw, bh, (255, 150, 50)),
        Element(1, "TOP-RIGHT", margin + bw + gap, top, bw, bh, (50, 200, 255)),
        Element(2, "BOT-LEFT", margin, top + bh + gap, bw, bh, (50, 255, 136)),
        Element(3, "BOT-RIGHT", margin + bw + gap, top + bh + gap, bw, bh, (200, 100, 255)),
    ]

def layout_six_grid(sw, sh):
    """3x2 grid — harder, closer to real UI."""
    margin = 50
    gap = 16
    cols, rows = 3, 2
    bw = (sw - 2 * margin - (cols - 1) * gap) // cols
    bh = (sh - 120 - gap) // rows
    top = 70
    colors = [(255,150,50), (200,50,200), (50,200,255),
              (50,255,136), (255,100,150), (200,200,50)]
    labels = ["EMAIL", "SEARCH", "SETTINGS", "PROFILE", "MESSAGES", "HELP"]
    elements = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x = margin + c * (bw + gap)
            y = top + r * (bh + gap)
            elements.append(Element(idx, labels[idx], x, y, bw, bh, colors[idx]))
    return elements

LAYOUTS = [
    ("2 BOXES", layout_two_boxes),
    ("4 QUADRANTS", layout_four_quadrants),
    ("6 GRID", layout_six_grid),
]


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1470, 956
    print(f"Screen: {sw}x{sh}")

    # ── EEG ──────────────────────────────────────────────────
    eeg = None
    if USE_EEG:
        if not is_bridge_running():
            print("WARNING: Muse bridge not running. Use --no-eeg or start bridge.")
            print("Continuing without EEG.")
            USE_EEG_local = False
        else:
            eeg = EEGBridgeClient()
            eeg.start()
            for _ in range(40):
                eeg.pull()
                time.sleep(0.05)
            print("EEG connected")
            USE_EEG_local = True
    else:
        USE_EEG_local = False
        print("Running without EEG (--no-eeg)")

    # ── Gaze calibration ─────────────────────────────────────
    cam_idx = 1  # MacBook built-in
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        ret, _ = cap.read()
        cap.release()
        if ret:
            cam_idx = idx
            break

    gaze = GazeEstimator(model_name="tiny_mlp")
    print("\n=== GAZE CALIBRATION ===")
    print("  Keep window focused. Move head slightly between dots.")
    run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                pulse_d=1.0, cd_d=1.0, camera_index=cam_idx)
    print("  Done.\n")

    # ── Smoothing ─────────────────────────────────────────────
    filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.008)
    filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.008)

    # ── Main loop ─────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Gaze-Select", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gaze-Select", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    engine = AttentionEngine()
    layout_idx = 0
    elements = LAYOUTS[layout_idx][1](sw, sh)
    selection_log = []
    gaze_x, gaze_y = sw // 2, sh // 2
    last_time = time.time()
    engagement = 0.5
    eng_history = deque(maxlen=15)

    print("  Controls: TAB=switch layout  R=recalibrate  ESC=quit")
    print(f"  Layout: {LAYOUTS[layout_idx][0]}")

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        # ── EEG ──────────────────────────────────────────────
        eeg_window = None
        if eeg and USE_EEG_local:
            eeg.pull()
            eeg_window = eeg.get_window(2.0)
            eng = compute_engagement(eeg_window)
            eng_history.append(eng)
            engagement = np.mean(eng_history)

        # ── Gaze ─────────────────────────────────────────────
        ret, frame = cap.read()
        if ret:
            features, blink = gaze.extract_features(frame)
            if features is not None and not blink:
                raw = gaze.predict(np.array([features]))[0]
                t = time.time()
                gaze_x = int(filter_x(raw[0], t))
                gaze_y = int(filter_y(raw[1], t))

        # ── Attention engine ─────────────────────────────────
        selected = engine.update(elements, gaze_x, gaze_y, engagement, dt)

        if selected >= 0:
            el = elements[selected]
            selection_log.append((now, el.label))
            print(f"  SELECTED: {el.label} (confidence={el.confidence:.0%})")
            # Reset all confidences after selection
            for e in elements:
                e.confidence = 0
                e.selected = False
            time.sleep(0.3)

        # ── Draw ─────────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (20, 6, 6)

        # Header
        cv2.putText(canvas, "GAZE-SELECT", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 208, 232), 1)
        cv2.putText(canvas, f"Layout: {LAYOUTS[layout_idx][0]}", (sw // 2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)
        cv2.putText(canvas, f"Selections: {len(selection_log)}", (sw - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)

        # Elements
        for el in elements:
            draw_element(canvas, el, now)

        # Gaze dot (subtle)
        cv2.circle(canvas, (gaze_x, gaze_y), 4, (0, 180, 255), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 6, (255, 255, 255), 1)

        # Footer
        eeg_label = f"EEG: {engagement:.2f}" if USE_EEG_local else "EEG: off"
        cv2.putText(canvas, eeg_label, (20, sh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 120), 1)
        cv2.putText(canvas, "TAB=layout  R=recalibrate  ESC=quit",
                    (sw - 340, sh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 90), 1)

        cv2.imshow("Gaze-Select", canvas)
        key = cv2.waitKey(16) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 9:  # TAB
            layout_idx = (layout_idx + 1) % len(LAYOUTS)
            elements = LAYOUTS[layout_idx][1](sw, sh)
            engine = AttentionEngine()
            print(f"  Layout: {LAYOUTS[layout_idx][0]}")
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                        pulse_d=1.0, cd_d=1.0, camera_index=cam_idx)
            filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.008)
            filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.008)
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Gaze-Select", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Gaze-Select", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    if eeg:
        eeg.stop()
    cv2.destroyAllWindows()

    print(f"\n  Total selections: {len(selection_log)}")
    for t, label in selection_log[-10:]:
        print(f"    {label}")
    print("\nDone.")


if __name__ == "__main__":
    main()
