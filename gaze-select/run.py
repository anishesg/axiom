#!/usr/bin/env python3
"""Gaze-Select v2: Intelligent element selection from coarse gaze.

Improvements over v1:
  - Anisotropic One Euro Filter (separate X/Y tuning)
  - Hysteresis + 150ms exit delay (kills flicker)
  - 90px invisible motor space expansion (biggest accuracy hack)
  - Adaptive thresholds by element density
  - Micro-animations (scale, glow pulse, spring)

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
# ONE EURO FILTER — anisotropic (separate X/Y tuning)
# ═════════════════════════════════════════════════════════════

class OneEuroFilter:
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
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ═════════════════════════════════════════════════════════════
# ELEMENT ZONE
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 90       # invisible hit area expansion (px)
HYSTERESIS_PAD = 20  # inner/outer boundary difference (px)
EXIT_DELAY = 0.15    # seconds before deactivating after gaze leaves

@dataclass
class Element:
    id: int
    label: str
    x: int
    y: int
    w: int
    h: int
    color: tuple

    # Live state
    confidence: float = 0.0
    dwell_time: float = 0.0
    gaze_in: bool = False
    highlighted: bool = False
    selected: bool = False
    last_enter: float = 0.0
    last_exit: float = 0.0
    stability: float = 0.0
    anim_scale: float = 1.0     # current visual scale
    anim_glow: float = 0.0      # glow pulse phase

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    def in_motor_zone(self, gx, gy):
        """Expanded invisible hit area (90px beyond visual boundary)."""
        return (self.x - MOTOR_PAD <= gx <= self.x + self.w + MOTOR_PAD and
                self.y - MOTOR_PAD <= gy <= self.y + self.h + MOTOR_PAD)

    def in_inner_zone(self, gx, gy):
        """Shrunk activation boundary (hysteresis inner)."""
        h = HYSTERESIS_PAD
        return (self.x + h <= gx <= self.x + self.w - h and
                self.y + h <= gy <= self.y + self.h - h)


# ═════════════════════════════════════════════════════════════
# ATTENTION ENGINE v2
# ═════════════════════════════════════════════════════════════

class AttentionEngine:
    def __init__(self, n_elements=2):
        self.dwell_weight = 0.4
        self.stability_weight = 0.3
        self.proximity_weight = 0.2
        self.eeg_weight = 0.1

        # Adaptive thresholds: dwell_ms = base * (1 + log2(N/4))
        base_dwell = 1.5
        self.dwell_saturate = base_dwell * (1 + max(0, math.log2(n_elements / 4)))
        self.highlight_threshold = 0.3
        self.select_threshold = 0.75
        self.decay_rate = 0.93

        self.gaze_history = deque(maxlen=20)
        self.active_id = -1

    def update(self, elements, gaze_x, gaze_y, engagement, dt):
        now = time.time()
        self.gaze_history.append((gaze_x, gaze_y, now))

        # Gaze velocity → stability
        gaze_speed = 0.0
        if len(self.gaze_history) >= 3:
            pts = list(self.gaze_history)
            dists = [math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                     for i in range(1, len(pts))]
            time_span = pts[-1][2] - pts[0][2]
            if time_span > 0:
                gaze_speed = sum(dists) / time_span
        stability = max(0, min(1, 1.0 - (gaze_speed - 50) / 450))

        selected_id = -1

        for el in elements:
            dist = math.hypot(gaze_x - el.cx, gaze_y - el.cy)

            # Hysteresis: use inner zone to activate, motor zone to stay active
            in_motor = el.in_motor_zone(gaze_x, gaze_y)
            in_inner = el.in_inner_zone(gaze_x, gaze_y)

            if el.gaze_in:
                # Already active — stay active if in motor zone
                if in_motor:
                    el.dwell_time = now - el.last_enter
                    el.stability = stability
                    el.last_exit = 0
                else:
                    # Gaze left motor zone — start exit delay
                    if el.last_exit == 0:
                        el.last_exit = now
                    if now - el.last_exit > EXIT_DELAY:
                        el.gaze_in = False
                        el.dwell_time = 0.0
                        el.stability = 0.0
                        el.last_exit = 0
            else:
                # Not active — require inner zone to activate (hysteresis)
                if in_inner:
                    el.last_enter = now
                    el.gaze_in = True
                    el.last_exit = 0
                    el.dwell_time = 0.0

            # Proximity: 1.0 at center, fading outward
            max_dist = math.hypot(el.w, el.h) / 2 + MOTOR_PAD
            proximity = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0

            # Confidence components
            dwell_score = min(1.0, el.dwell_time / self.dwell_saturate)
            stability_score = el.stability if el.gaze_in else 0
            proximity_score = proximity
            eeg_score = max(0, min(1, engagement))

            if el.gaze_in:
                raw = (self.dwell_weight * dwell_score +
                       self.stability_weight * stability_score +
                       self.proximity_weight * proximity_score +
                       self.eeg_weight * eeg_score)
                el.confidence = el.confidence * 0.7 + raw * 0.3
            else:
                el.confidence *= self.decay_rate

            el.confidence = max(0, min(1, el.confidence))
            el.highlighted = el.confidence >= self.highlight_threshold

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
        return 0.5
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
    bp1 = bp(eeg_window[1])
    bp2 = bp(eeg_window[2])
    alpha = (bp1["alpha"] + bp2["alpha"]) / 2
    theta = (bp1["theta"] + bp2["theta"]) / 2
    beta = (bp1["beta"] + bp2["beta"]) / 2
    return beta / (alpha + theta + 0.001)


# ═════════════════════════════════════════════════════════════
# RENDERING v2 — micro-animations
# ═════════════════════════════════════════════════════════════

def draw_element(canvas, el: Element, now: float):
    c = el.confidence

    # Animate scale: spring toward target
    target_scale = 1.0 + c * 0.04  # max 1.04 at full confidence
    el.anim_scale += (target_scale - el.anim_scale) * 0.15  # spring

    # Animate glow pulse (0.42Hz heartbeat when highlighted)
    if c >= 0.3:
        el.anim_glow += 0.042  # ~0.42Hz at 60fps
    else:
        el.anim_glow *= 0.9

    glow_intensity = 0.5 + 0.5 * math.sin(el.anim_glow * 2 * math.pi)

    # Compute scaled rect
    scale = el.anim_scale
    sw = int(el.w * scale)
    sh = int(el.h * scale)
    sx = el.cx - sw // 2
    sy = el.cy - sh // 2

    # Background fill
    if c > 0.01:
        overlay = canvas.copy()
        fill_alpha = 0.05 + c * 0.2
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), el.color, -1)
        cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

    # Border with glow
    if c >= 0.75:
        glow_c = int(200 + 55 * glow_intensity)
        cv2.rectangle(canvas, (sx - 1, sy - 1), (sx + sw + 1, sy + sh + 1),
                      (glow_c, glow_c, glow_c), 3)
    elif c >= 0.3:
        gc = tuple(int(v * (0.7 + 0.3 * glow_intensity)) for v in el.color)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), gc, 2)
    else:
        dim = tuple(int(v * 0.25) for v in el.color)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), dim, 1)

    # Label
    label_alpha = 0.3 + c * 0.7
    label_color = tuple(int(v * label_alpha) for v in (255, 255, 255))
    ts = cv2.getTextSize(el.label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(canvas, el.label,
                (el.cx - ts[0] // 2, el.cy + ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # Confidence ring
    if c >= 0.3:
        radius = 28
        angle = int(360 * min(1, c / 0.75))
        ring_color = (0, 255, 136) if c >= 0.6 else el.color
        cv2.ellipse(canvas, (el.cx, el.cy - 45), (radius, radius),
                    -90, 0, 360, (35, 35, 55), 2)
        cv2.ellipse(canvas, (el.cx, el.cy - 45), (radius, radius),
                    -90, 0, angle, ring_color, 3)
        # Percentage inside ring
        pct = f"{c:.0%}"
        pts = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.putText(canvas, pct, (el.cx - pts[0] // 2, el.cy - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ring_color, 1)

    # Selection flash (green burst, fades)
    if el.selected:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (0, 255, 136), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)


# ═════════════════════════════════════════════════════════════
# LAYOUTS
# ═════════════════════════════════════════════════════════════

def layout_two_boxes(sw, sh):
    gap = sw // 5
    bw = (sw - 3 * gap) // 2
    bh = sh // 2
    by = (sh - bh) // 2 + 20
    return [
        Element(0, "LEFT", gap, by, bw, bh, (255, 150, 50)),
        Element(1, "RIGHT", 2 * gap + bw, by, bw, bh, (50, 200, 255)),
    ]

def layout_four_quadrants(sw, sh):
    margin = 60
    gap = 24
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
    margin = 50
    gap = 20
    cols, rows = 3, 2
    bw = (sw - 2 * margin - (cols - 1) * gap) // cols
    bh = (sh - 130 - gap) // rows
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

def layout_nine_grid(sw, sh):
    """3x3 — stress test for density."""
    margin = 40
    gap = 14
    cols, rows = 3, 3
    bw = (sw - 2 * margin - (cols - 1) * gap) // cols
    bh = (sh - 120 - (rows - 1) * gap) // rows
    top = 65
    colors = [(255,150,50), (200,50,200), (50,200,255),
              (50,255,136), (255,100,150), (200,200,50),
              (150,200,255), (255,200,100), (100,255,200)]
    labels = ["INBOX", "COMPOSE", "SEARCH", "STARRED", "SENT", "DRAFTS",
              "ARCHIVE", "TRASH", "SETTINGS"]
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
    ("9 GRID", layout_nine_grid),
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
    USE_EEG_local = False
    if USE_EEG:
        if not is_bridge_running():
            print("WARNING: Muse bridge not running. Continuing without EEG.")
        else:
            eeg = EEGBridgeClient()
            eeg.start()
            for _ in range(40):
                eeg.pull()
                time.sleep(0.05)
            print("EEG connected")
            USE_EEG_local = True
    else:
        print("Running without EEG (--no-eeg)")

    # ── Gaze calibration ─────────────────────────────────────
    cam_idx = 1
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

    # ── Anisotropic smoothing ────────────────────────────────
    # X: faster saccades, needs more responsiveness
    # Y: slower, noisier from head pitch, needs more smoothing
    filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)

    # ── Main loop ────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Gaze-Select", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gaze-Select", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    layout_idx = 0
    elements = LAYOUTS[layout_idx][1](sw, sh)
    engine = AttentionEngine(n_elements=len(elements))
    selection_log = []
    gaze_x, gaze_y = sw // 2, sh // 2
    last_time = time.time()
    engagement = 0.5
    eng_history = deque(maxlen=15)
    cooldown_until = 0.0

    print("  Controls: TAB=layout  R=recalibrate  ESC=quit")
    print(f"  Layout: {LAYOUTS[layout_idx][0]} ({len(elements)} elements)")
    print(f"  Dwell saturate: {engine.dwell_saturate:.1f}s")

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        # ── EEG ──────────────────────────────────────────────
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
        if now < cooldown_until:
            selected = -1
        else:
            selected = engine.update(elements, gaze_x, gaze_y, engagement, dt)

        if selected >= 0:
            el = elements[selected]
            selection_log.append((now, el.label))
            print(f"  SELECTED: {el.label} (confidence={el.confidence:.0%})")
            for e in elements:
                e.confidence = 0
                e.selected = False
                e.gaze_in = False
                e.anim_scale = 1.0
            cooldown_until = now + 1.5
            time.sleep(0.2)

        # ── Draw ─────────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (20, 6, 6)

        # Header
        cv2.putText(canvas, "GAZE-SELECT", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 208, 232), 1)
        layout_label = f"Layout: {LAYOUTS[layout_idx][0]}"
        cv2.putText(canvas, layout_label, (sw // 2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)
        cv2.putText(canvas, f"Selections: {len(selection_log)}", (sw - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)

        # Cooldown indicator
        if now < cooldown_until:
            remaining = cooldown_until - now
            cv2.putText(canvas, f"COOLDOWN {remaining:.1f}s", (sw // 2 - 60, sh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 120), 1)

        # Elements
        for el in elements:
            draw_element(canvas, el, now)

        # Gaze dot
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

        if key == 27:
            break
        elif key == 9:  # TAB
            layout_idx = (layout_idx + 1) % len(LAYOUTS)
            elements = LAYOUTS[layout_idx][1](sw, sh)
            engine = AttentionEngine(n_elements=len(elements))
            cooldown_until = 0
            print(f"  Layout: {LAYOUTS[layout_idx][0]} ({len(elements)} elements, "
                  f"dwell={engine.dwell_saturate:.1f}s)")
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                        pulse_d=1.0, cd_d=1.0, camera_index=cam_idx)
            filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
            filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)
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
