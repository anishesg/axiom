#!/usr/bin/env python3
"""Axiom v3 — Gaze + EEG element selection.

Two panels (LEFT / RIGHT), each containing clickable elements.
Gaze determines which element you're looking at.
EEG engagement + dwell determines if you intend to click it.
Progressive build-up → selection.

Usage:
    python3 run.py              # with Muse EEG
    python3 run.py --no-eeg     # gaze only
"""

import cv2
import math
import numpy as np
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from screeninfo import get_monitors

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
# ONE EURO FILTER
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
# ELEMENTS
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 60
HYSTERESIS_PAD = 12
EXIT_DELAY = 0.15

@dataclass
class Element:
    id: int
    label: str
    x: int
    y: int
    w: int
    h: int
    color: tuple
    panel: str  # "left" or "right"

    confidence: float = 0.0
    dwell_time: float = 0.0
    gaze_in: bool = False
    last_enter: float = 0.0
    last_exit: float = 0.0
    stability: float = 0.0
    anim_scale: float = 1.0
    anim_glow: float = 0.0

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2


def build_elements(sw, sh):
    """Two panels, each with distinct clickable elements."""
    elements = []
    i = 0

    panel_w = (sw - 60) // 2
    left_x = 20
    right_x = 40 + panel_w

    # Left panel elements
    left_items = [
        ("Inbox", (255, 180, 50)),
        ("Compose", (50, 220, 130)),
        ("Search", (80, 160, 255)),
        ("Settings", (200, 100, 255)),
    ]
    item_h = 70
    gap = 14
    start_y = 100
    for j, (label, color) in enumerate(left_items):
        y = start_y + j * (item_h + gap)
        elements.append(Element(i, label, left_x + 20, y, panel_w - 40, item_h, color, "left"))
        i += 1

    # Right panel elements
    right_items = [
        ("Profile", (255, 100, 150)),
        ("Messages", (50, 200, 255)),
        ("Notifications", (255, 200, 50)),
        ("Logout", (200, 60, 60)),
    ]
    for j, (label, color) in enumerate(right_items):
        y = start_y + j * (item_h + gap)
        elements.append(Element(i, label, right_x + 20, y, panel_w - 40, item_h, color, "right"))
        i += 1

    return elements, panel_w, left_x, right_x


# ═════════════════════════════════════════════════════════════
# ATTENTION ENGINE
# ═════════════════════════════════════════════════════════════

class AttentionEngine:
    def __init__(self, n_elements):
        self.dwell_weight = 0.35
        self.stability_weight = 0.30
        self.proximity_weight = 0.15
        self.eeg_weight = 0.20

        self.dwell_saturate = 1.5 * (1 + max(0, math.log2(n_elements / 4)))
        self.select_threshold = 0.75
        self.decay_rate = 0.91
        self.gaze_history = deque(maxlen=20)

    def update(self, elements, gaze_x, gaze_y, engagement, dt):
        now = time.time()
        self.gaze_history.append((gaze_x, gaze_y, now))

        gaze_speed = 0.0
        if len(self.gaze_history) >= 3:
            pts = list(self.gaze_history)
            dists = [math.hypot(pts[k][0] - pts[k-1][0], pts[k][1] - pts[k-1][1])
                     for k in range(1, len(pts))]
            span = pts[-1][2] - pts[0][2]
            if span > 0:
                gaze_speed = sum(dists) / span
        stability = max(0, min(1, 1.0 - (gaze_speed - 40) / 400))

        selected_id = -1

        for el in elements:
            dist = math.hypot(gaze_x - el.cx, gaze_y - el.cy)

            in_motor = (el.x - MOTOR_PAD <= gaze_x <= el.x + el.w + MOTOR_PAD and
                        el.y - MOTOR_PAD <= gaze_y <= el.y + el.h + MOTOR_PAD)
            in_inner = (el.x + HYSTERESIS_PAD <= gaze_x <= el.x + el.w - HYSTERESIS_PAD and
                        el.y + HYSTERESIS_PAD <= gaze_y <= el.y + el.h - HYSTERESIS_PAD)

            if el.gaze_in:
                if in_motor:
                    el.dwell_time = now - el.last_enter
                    el.stability = stability
                    el.last_exit = 0
                else:
                    if el.last_exit == 0:
                        el.last_exit = now
                    if now - el.last_exit > EXIT_DELAY:
                        el.gaze_in = False
                        el.dwell_time = 0
                        el.stability = 0
                        el.last_exit = 0
            else:
                if in_inner:
                    el.last_enter = now
                    el.gaze_in = True
                    el.last_exit = 0
                    el.dwell_time = 0

            max_dist = math.hypot(el.w, el.h) / 2 + MOTOR_PAD
            proximity = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0

            dwell_score = min(1.0, el.dwell_time / self.dwell_saturate)
            stab_score = el.stability if el.gaze_in else 0
            prox_score = proximity
            eeg_score = max(0, min(1, engagement))

            if el.gaze_in:
                raw = (self.dwell_weight * dwell_score +
                       self.stability_weight * stab_score +
                       self.proximity_weight * prox_score +
                       self.eeg_weight * eeg_score)
                el.confidence = el.confidence * 0.7 + raw * 0.3
            else:
                el.confidence *= self.decay_rate

            el.confidence = max(0, min(1, el.confidence))

            if el.confidence >= self.select_threshold:
                selected_id = el.id

        return selected_id


# ═════════════════════════════════════════════════════════════
# EEG
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
# RENDERING
# ═════════════════════════════════════════════════════════════

def draw_element(canvas, el: Element, now):
    c = el.confidence

    target_scale = 1.0 + c * 0.03
    el.anim_scale += (target_scale - el.anim_scale) * 0.15

    if c >= 0.25:
        el.anim_glow += 0.042
    else:
        el.anim_glow *= 0.9

    glow = 0.5 + 0.5 * math.sin(el.anim_glow * 2 * math.pi)

    sc = el.anim_scale
    sw = int(el.w * sc)
    sh = int(el.h * sc)
    sx = el.cx - sw // 2
    sy = el.cy - sh // 2

    # Fill
    if c > 0.01:
        ov = canvas.copy()
        fa = 0.04 + c * 0.2
        cv2.rectangle(ov, (sx, sy), (sx + sw, sy + sh), el.color, -1)
        cv2.addWeighted(ov, fa, canvas, 1 - fa, 0, canvas)

    # Border
    if c >= 0.65:
        gc = int(200 + 55 * glow)
        cv2.rectangle(canvas, (sx-1, sy-1), (sx+sw+1, sy+sh+1), (gc, gc, gc), 3)
    elif c >= 0.25:
        gc = tuple(int(v * (0.6 + 0.4 * glow)) for v in el.color)
        cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), gc, 2)
    else:
        dim = tuple(int(v * 0.2) for v in el.color)
        cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), dim, 1)

    # Label
    la = 0.25 + c * 0.75
    lc = tuple(int(255 * la) for _ in range(3))
    ts = cv2.getTextSize(el.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(canvas, el.label, (el.cx - ts[0]//2, el.cy + ts[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)

    # Ring
    if c >= 0.25:
        r = 22
        angle = int(360 * min(1, c / 0.75))
        rc = (0, 255, 136) if c >= 0.55 else el.color
        cv2.ellipse(canvas, (el.cx, el.cy - 32), (r, r), -90, 0, 360, (30, 30, 50), 2)
        cv2.ellipse(canvas, (el.cx, el.cy - 32), (r, r), -90, 0, angle, rc, 3)
        pct = f"{c:.0%}"
        ps = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)[0]
        cv2.putText(canvas, pct, (el.cx - ps[0]//2, el.cy - 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, rc, 1)

    # Selection flash
    if c >= 0.75:
        ov = canvas.copy()
        cv2.rectangle(ov, (sx, sy), (sx+sw, sy+sh), (0, 255, 136), -1)
        cv2.addWeighted(ov, 0.3, canvas, 0.7, 0, canvas)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    try:
        m = get_monitors()[0]
        scr_w, scr_h = m.width, m.height
    except Exception:
        scr_w, scr_h = 1470, 956
    print(f"Screen: {scr_w}x{scr_h}")

    # EEG
    eeg = None
    USE_EEG_local = False
    if USE_EEG:
        if not is_bridge_running():
            print("WARNING: No Muse bridge. Continuing without EEG.")
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

    # Gaze
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

    filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)

    # Build elements
    elements, panel_w, left_x, right_x = build_elements(scr_w, scr_h)
    engine = AttentionEngine(n_elements=len(elements))
    selection_log = []
    gaze_x, gaze_y = scr_w // 2, scr_h // 2
    last_time = time.time()
    engagement = 0.5
    eng_history = deque(maxlen=15)
    cooldown_until = 0.0

    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Axiom v3", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Axiom v3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("  8 elements across 2 panels. Look + focus to select.")
    print("  ESC=quit  R=recalibrate\n")

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        # EEG
        if eeg and USE_EEG_local:
            eeg.pull()
            w = eeg.get_window(2.0)
            eng = compute_engagement(w)
            eng_history.append(eng)
            engagement = np.mean(eng_history)

        # Gaze
        ret, frame = cap.read()
        if ret:
            features, blink = gaze.extract_features(frame)
            if features is not None and not blink:
                raw = gaze.predict(np.array([features]))[0]
                t = time.time()
                gaze_x = int(filter_x(raw[0], t))
                gaze_y = int(filter_y(raw[1], t))

        # Attention
        if now < cooldown_until:
            selected = -1
        else:
            selected = engine.update(elements, gaze_x, gaze_y, engagement, dt)

        if selected >= 0:
            el = elements[selected]
            selection_log.append((now, el.label, el.panel))
            print(f"  SELECTED: {el.label} [{el.panel}] (confidence={el.confidence:.0%})")
            for e in elements:
                e.confidence = 0
                e.gaze_in = False
                e.anim_scale = 1.0
            cooldown_until = now + 1.5
            time.sleep(0.2)

        # ── Draw ─────────────────────────────────────────────
        canvas = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
        canvas[:] = (18, 5, 5)

        # Panel backgrounds
        cv2.rectangle(canvas, (left_x, 60), (left_x + panel_w, scr_h - 60),
                      (25, 25, 40), -1)
        cv2.rectangle(canvas, (right_x, 60), (right_x + panel_w, scr_h - 60),
                      (25, 25, 40), -1)

        # Panel labels
        cv2.putText(canvas, "LEFT PANEL", (left_x + panel_w // 2 - 50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 100), 1)
        cv2.putText(canvas, "RIGHT PANEL", (right_x + panel_w // 2 - 55, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 100), 1)

        # Header
        cv2.putText(canvas, "AXIOM v3", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 208, 232), 1)
        cv2.putText(canvas, f"Selections: {len(selection_log)}", (scr_w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)

        # Elements
        for el in elements:
            draw_element(canvas, el, now)

        # Gaze dot
        cv2.circle(canvas, (gaze_x, gaze_y), 4, (0, 180, 255), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 6, (255, 255, 255), 1)

        # Engagement bar (bottom)
        bar_x, bar_y = 40, scr_h - 45
        bar_w, bar_h = scr_w - 80, 16
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (30, 30, 50), -1)
        eng_norm = max(0, min(1, engagement))
        fill_w = int(bar_w * eng_norm)
        eng_color = (0, 255, 136) if engagement > 0.5 else (0, 160, 255)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      eng_color, -1)
        cv2.putText(canvas, f"Engagement: {engagement:.2f}",
                    (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 160), 1)

        if now < cooldown_until:
            cv2.putText(canvas, f"COOLDOWN {cooldown_until - now:.1f}s",
                        (scr_w // 2 - 50, scr_h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 120), 1)

        eeg_label = "EEG: LIVE" if USE_EEG_local else "EEG: off"
        cv2.putText(canvas, eeg_label, (20, scr_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 90), 1)
        cv2.putText(canvas, "R=recalibrate  ESC=quit", (scr_w - 220, scr_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 70), 1)

        cv2.imshow("Axiom v3", canvas)
        key = cv2.waitKey(16) & 0xFF

        if key == 27:
            break
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                        pulse_d=1.0, cd_d=1.0, camera_index=cam_idx)
            filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
            filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Axiom v3", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Axiom v3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    if eeg:
        eeg.stop()
    cv2.destroyAllWindows()

    print(f"\n  Total: {len(selection_log)} selections")
    for t, label, panel in selection_log:
        print(f"    [{panel:>5s}] {label}")
    print("\nDone.")


if __name__ == "__main__":
    main()
