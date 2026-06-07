#!/usr/bin/env python3
"""Intent Engine: Bayesian gaze + EEG intent inference.

Instead of "where are your eyes pointing", this asks:
"what do you INTEND to do, given your gaze, brain state, and page context?"

Architecture:
  1. Page layout → semantic zone map (nav, content, actions, etc.)
  2. Each zone has a PRIOR probability based on context
  3. Gaze + EEG + dwell + stability update the posterior per-frame
  4. When P(intent=act_on_zone) crosses threshold → select

The test simulates a simple web page with semantic zones:
  - Navigation bar (top)
  - Content area (center)
  - Action buttons (bottom)
  - Sidebar links

Each zone type has different priors and thresholds.

Usage:
    python3 run.py --no-eeg     # gaze only
    python3 run.py              # with Muse EEG
"""

import cv2
import math
import numpy as np
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

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
# SEMANTIC ZONE TYPES — each behaves differently
# ═════════════════════════════════════════════════════════════

class ZoneType(Enum):
    NAV = "nav"           # navigation — quick actions, low dwell needed
    CONTENT = "content"   # reading area — high dwell expected, don't trigger
    ACTION = "action"     # buttons/CTAs — moderate dwell, high intent needed
    LINK = "link"         # clickable links — moderate everything
    PASSIVE = "passive"   # decorative/non-interactive — never triggers


# Per-zone-type behavior tuning
ZONE_CONFIG = {
    ZoneType.NAV: {
        "base_prior": 0.15,       # moderate prior — nav is common
        "dwell_to_select": 1.2,   # fast — nav clicks are quick
        "engagement_boost": 0.1,  # slight EEG boost helps
        "stability_required": 0.4,
        "color": (255, 180, 50),
    },
    ZoneType.CONTENT: {
        "base_prior": 0.05,       # low prior — reading ≠ clicking
        "dwell_to_select": 4.0,   # very long — reading dwells are normal
        "engagement_boost": 0.3,  # needs strong engagement signal to trigger
        "stability_required": 0.6,
        "color": (80, 80, 100),
    },
    ZoneType.ACTION: {
        "base_prior": 0.20,       # high prior — buttons are meant to be clicked
        "dwell_to_select": 1.5,   # moderate
        "engagement_boost": 0.2,
        "stability_required": 0.5,
        "color": (50, 220, 130),
    },
    ZoneType.LINK: {
        "base_prior": 0.12,
        "dwell_to_select": 1.8,
        "engagement_boost": 0.15,
        "stability_required": 0.45,
        "color": (255, 130, 80),
    },
    ZoneType.PASSIVE: {
        "base_prior": 0.0,
        "dwell_to_select": 99.0,
        "engagement_boost": 0.0,
        "stability_required": 1.0,
        "color": (40, 40, 55),
    },
}


@dataclass
class Zone:
    id: int
    label: str
    zone_type: ZoneType
    x: int
    y: int
    w: int
    h: int

    # Bayesian state
    prior: float = 0.0
    posterior: float = 0.0
    likelihood: float = 0.0

    # Gaze state
    gaze_in: bool = False
    dwell_time: float = 0.0
    last_enter: float = 0.0
    last_exit: float = 0.0
    stability: float = 0.0

    # Animation
    anim_scale: float = 1.0
    anim_glow: float = 0.0

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def config(self):
        return ZONE_CONFIG[self.zone_type]

    @property
    def color(self):
        return self.config["color"]


# ═════════════════════════════════════════════════════════════
# INTENT ENGINE — Bayesian inference over zones
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 80
HYSTERESIS_PAD = 15
EXIT_DELAY = 0.15

class IntentEngine:
    """Bayesian intent inference: P(act_on_zone | gaze, eeg, context)."""

    def __init__(self, zones: list[Zone]):
        self.zones = zones
        self.gaze_history = deque(maxlen=20)
        self.selection_history = []  # (time, zone_id) for context priors
        self.select_threshold = 0.80

        # Initialize priors from zone types
        self._reset_priors()

    def _reset_priors(self):
        total = sum(z.config["base_prior"] for z in self.zones) + 0.001
        for z in self.zones:
            z.prior = z.config["base_prior"] / total

    def _update_context_priors(self):
        """Adjust priors based on selection history.
        If user just clicked NAV, next click is likely CONTENT or ACTION."""
        if not self.selection_history:
            return

        last_zone_id = self.selection_history[-1][1]
        last_zone = next((z for z in self.zones if z.id == last_zone_id), None)
        if not last_zone:
            return

        # Context transitions: what's likely after each zone type?
        transition_boosts = {
            ZoneType.NAV: {ZoneType.CONTENT: 1.5, ZoneType.ACTION: 1.3},
            ZoneType.CONTENT: {ZoneType.ACTION: 1.8, ZoneType.NAV: 1.2},
            ZoneType.ACTION: {ZoneType.NAV: 1.4, ZoneType.CONTENT: 1.3},
            ZoneType.LINK: {ZoneType.CONTENT: 1.5, ZoneType.NAV: 1.2},
        }

        boosts = transition_boosts.get(last_zone.zone_type, {})
        for z in self.zones:
            if z.zone_type in boosts:
                z.prior *= boosts[z.zone_type]

        # Renormalize
        total = sum(z.prior for z in self.zones) + 0.001
        for z in self.zones:
            z.prior /= total

    def update(self, gaze_x, gaze_y, engagement, dt):
        now = time.time()
        self.gaze_history.append((gaze_x, gaze_y, now))

        # Gaze velocity → stability
        gaze_speed = 0.0
        if len(self.gaze_history) >= 3:
            pts = list(self.gaze_history)
            dists = [math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
                     for i in range(1, len(pts))]
            span = pts[-1][2] - pts[0][2]
            if span > 0:
                gaze_speed = sum(dists) / span
        stability = max(0, min(1, 1.0 - (gaze_speed - 40) / 400))

        selected_id = -1

        for z in self.zones:
            if z.zone_type == ZoneType.PASSIVE:
                z.posterior = 0
                continue

            # Hit test with hysteresis
            in_motor = (z.x - MOTOR_PAD <= gaze_x <= z.x + z.w + MOTOR_PAD and
                        z.y - MOTOR_PAD <= gaze_y <= z.y + z.h + MOTOR_PAD)
            in_inner = (z.x + HYSTERESIS_PAD <= gaze_x <= z.x + z.w - HYSTERESIS_PAD and
                        z.y + HYSTERESIS_PAD <= gaze_y <= z.y + z.h - HYSTERESIS_PAD)

            if z.gaze_in:
                if in_motor:
                    z.dwell_time = now - z.last_enter
                    z.stability = stability
                    z.last_exit = 0
                else:
                    if z.last_exit == 0:
                        z.last_exit = now
                    if now - z.last_exit > EXIT_DELAY:
                        z.gaze_in = False
                        z.dwell_time = 0
                        z.stability = 0
                        z.last_exit = 0
            else:
                if in_inner:
                    z.last_enter = now
                    z.gaze_in = True
                    z.last_exit = 0
                    z.dwell_time = 0

            # ── Compute likelihood P(observations | intent=act_on_z) ──

            cfg = z.config

            # Dwell component: how long vs expected for this zone type
            dwell_score = min(1.0, z.dwell_time / cfg["dwell_to_select"])

            # Stability component: must exceed zone's requirement
            stab_score = max(0, z.stability - cfg["stability_required"] + 0.5)
            stab_score = min(1.0, stab_score)

            # Proximity to center
            dist = math.hypot(gaze_x - z.cx, gaze_y - z.cy)
            max_dist = math.hypot(z.w, z.h) / 2 + MOTOR_PAD
            proximity = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0

            # EEG engagement boost (zone-type-specific weight)
            eeg_component = max(0, min(1, engagement)) * cfg["engagement_boost"]

            if z.gaze_in:
                # Likelihood = weighted combination
                z.likelihood = (0.35 * dwell_score +
                                0.25 * stab_score +
                                0.20 * proximity +
                                0.20 * eeg_component)

                # Bayesian update: posterior ∝ prior × likelihood
                raw_posterior = z.prior * (0.5 + z.likelihood)
                z.posterior = z.posterior * 0.7 + raw_posterior * 0.3
            else:
                z.likelihood = 0
                z.posterior *= 0.90  # decay

            z.posterior = max(0, min(1, z.posterior))

            # Animate
            target_scale = 1.0 + z.posterior * 0.04
            z.anim_scale += (target_scale - z.anim_scale) * 0.15
            if z.posterior >= 0.3:
                z.anim_glow += 0.042
            else:
                z.anim_glow *= 0.9

            # Selection
            if z.posterior >= self.select_threshold:
                selected_id = z.id

        # Normalize posteriors (they should sum to ≤ 1)
        total_post = sum(z.posterior for z in self.zones) + 0.001
        if total_post > 1.0:
            for z in self.zones:
                z.posterior /= total_post

        return selected_id

    def on_selection(self, zone_id):
        self.selection_history.append((time.time(), zone_id))
        for z in self.zones:
            z.posterior = 0
            z.gaze_in = False
            z.dwell_time = 0
            z.anim_scale = 1.0
        self._reset_priors()
        self._update_context_priors()


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

def draw_zone(canvas, z: Zone, now: float, is_top_intent: bool):
    p = z.posterior
    cfg = z.config
    color = z.color

    glow = 0.5 + 0.5 * math.sin(z.anim_glow * 2 * math.pi)

    # Scaled rect
    sc = z.anim_scale
    sw = int(z.w * sc)
    sh = int(z.h * sc)
    sx = z.cx - sw // 2
    sy = z.cy - sh // 2

    # Fill
    if p > 0.01:
        overlay = canvas.copy()
        fill_a = 0.04 + p * 0.22
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), color, -1)
        cv2.addWeighted(overlay, fill_a, canvas, 1 - fill_a, 0, canvas)

    # Border
    if p >= 0.7:
        gc = int(200 + 55 * glow)
        cv2.rectangle(canvas, (sx-1, sy-1), (sx+sw+1, sy+sh+1), (gc, gc, gc), 3)
    elif p >= 0.3:
        gc = tuple(int(v * (0.6 + 0.4 * glow)) for v in color)
        cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), gc, 2)
    else:
        dim = tuple(int(v * 0.2) for v in color)
        cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), dim, 1)

    # Zone type tag (small, top-left corner)
    tag = z.zone_type.value.upper()
    cv2.putText(canvas, tag, (z.x + 6, z.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, tuple(int(v * 0.5) for v in color), 1)

    # Label
    alpha_f = 0.3 + p * 0.7
    lc = tuple(int(255 * alpha_f) for _ in range(3))
    ts = cv2.getTextSize(z.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(canvas, z.label, (z.cx - ts[0]//2, z.cy + ts[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)

    # Confidence ring + percentage
    if p >= 0.2:
        r = 24
        angle = int(360 * min(1, p / 0.8))
        rc = (0, 255, 136) if p >= 0.6 else color
        cv2.ellipse(canvas, (z.cx, z.cy - 38), (r, r), -90, 0, 360, (30, 30, 50), 2)
        cv2.ellipse(canvas, (z.cx, z.cy - 38), (r, r), -90, 0, angle, rc, 3)
        pct = f"{p:.0%}"
        ps = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(canvas, pct, (z.cx - ps[0]//2, z.cy - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, rc, 1)

    # Prior indicator (tiny bar at bottom)
    prior_w = int(z.w * z.prior * 3)
    cv2.rectangle(canvas, (z.x, z.y + z.h - 4), (z.x + min(prior_w, z.w), z.y + z.h - 1),
                  tuple(int(v * 0.4) for v in color), -1)

    # Selection flash
    if p >= 0.8:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sx, sy), (sx+sw, sy+sh), (0, 255, 136), -1)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)


# ═════════════════════════════════════════════════════════════
# PAGE LAYOUTS — simulate real web pages
# ═════════════════════════════════════════════════════════════

def page_email_client(sw, sh):
    """Simulated email client layout."""
    zones = []
    i = 0

    # Top nav bar
    nav_h = 50
    nav_items = ["Inbox", "Compose", "Search"]
    nav_w = (sw - 40) // len(nav_items)
    for j, label in enumerate(nav_items):
        zones.append(Zone(i, label, ZoneType.NAV,
                          20 + j * nav_w, 10, nav_w - 8, nav_h))
        i += 1

    # Sidebar
    sidebar_w = 160
    sidebar_items = ["Starred", "Sent", "Drafts", "Trash"]
    for j, label in enumerate(sidebar_items):
        zones.append(Zone(i, label, ZoneType.LINK,
                          20, 80 + j * 55, sidebar_w, 45))
        i += 1

    # Email list (content area)
    content_x = 200
    content_w = sw - 240
    email_h = 65
    emails = ["Meeting Tomorrow", "Project Update", "Weekly Report",
              "Lunch Plans", "Code Review"]
    for j, label in enumerate(emails):
        zones.append(Zone(i, label, ZoneType.LINK,
                          content_x, 80 + j * (email_h + 8), content_w, email_h))
        i += 1

    # Action buttons (bottom)
    btn_y = sh - 70
    btn_w = 140
    btns = ["Archive", "Delete", "Reply"]
    start_x = content_x
    for j, label in enumerate(btns):
        zones.append(Zone(i, label, ZoneType.ACTION,
                          start_x + j * (btn_w + 16), btn_y, btn_w, 50))
        i += 1

    return zones


def page_shopping(sw, sh):
    """Simulated product page."""
    zones = []
    i = 0

    # Nav
    for j, label in enumerate(["Home", "Cart", "Account"]):
        zones.append(Zone(i, label, ZoneType.NAV,
                          20 + j * 160, 10, 150, 45))
        i += 1

    # Product cards (2x2)
    card_w = (sw - 80) // 2
    card_h = (sh - 200) // 2
    products = ["Headphones $79", "Keyboard $129", "Mouse $49", "Monitor $299"]
    for r in range(2):
        for c in range(2):
            idx = r * 2 + c
            zones.append(Zone(i, products[idx], ZoneType.LINK,
                              30 + c * (card_w + 20), 80 + r * (card_h + 16),
                              card_w, card_h))
            i += 1

    # Buy button
    zones.append(Zone(i, "BUY NOW", ZoneType.ACTION,
                      sw // 2 - 100, sh - 70, 200, 55))
    i += 1

    return zones


def page_article(sw, sh):
    """Simulated article/blog page."""
    zones = []
    i = 0

    # Nav
    for j, label in enumerate(["Home", "Articles", "About"]):
        zones.append(Zone(i, label, ZoneType.NAV,
                          20 + j * 150, 10, 140, 40))
        i += 1

    # Article title (content — reading, not clicking)
    zones.append(Zone(i, "Article Title", ZoneType.CONTENT,
                      40, 70, sw - 80, 60))
    i += 1

    # Article body paragraphs
    for j in range(3):
        zones.append(Zone(i, f"Paragraph {j+1}", ZoneType.CONTENT,
                          40, 150 + j * 140, sw - 300, 120))
        i += 1

    # Sidebar links
    sidebar_x = sw - 220
    for j, label in enumerate(["Related 1", "Related 2", "Related 3"]):
        zones.append(Zone(i, label, ZoneType.LINK,
                          sidebar_x, 150 + j * 80, 200, 60))
        i += 1

    # Share / Comment buttons
    zones.append(Zone(i, "Share", ZoneType.ACTION,
                      40, sh - 70, 120, 50))
    i += 1
    zones.append(Zone(i, "Comment", ZoneType.ACTION,
                      180, sh - 70, 140, 50))
    i += 1

    return zones


PAGES = [
    ("EMAIL CLIENT", page_email_client),
    ("SHOPPING", page_shopping),
    ("ARTICLE", page_article),
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

    # Gaze calibration
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

    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Intent Engine", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Intent Engine", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    page_idx = 0
    zones = PAGES[page_idx][1](sw, sh)
    engine = IntentEngine(zones)
    selection_log = []
    gaze_x, gaze_y = sw // 2, sh // 2
    last_time = time.time()
    engagement = 0.5
    eng_history = deque(maxlen=15)
    cooldown_until = 0.0

    print(f"  Page: {PAGES[page_idx][0]} ({len(zones)} zones)")
    print("  Controls: TAB=page  R=recalibrate  ESC=quit\n")

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

        # Intent engine
        if now < cooldown_until:
            selected = -1
        else:
            selected = engine.update(gaze_x, gaze_y, engagement, dt)

        if selected >= 0:
            z = next(z for z in zones if z.id == selected)
            selection_log.append((now, z.label, z.zone_type.value))
            print(f"  INTENT: {z.label} [{z.zone_type.value}] "
                  f"(posterior={z.posterior:.0%}, prior={z.prior:.2f})")
            engine.on_selection(selected)
            cooldown_until = now + 1.5
            time.sleep(0.2)

        # ── Draw ─────────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (18, 5, 5)

        # Header
        cv2.putText(canvas, "INTENT ENGINE", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 208, 232), 1)
        cv2.putText(canvas, f"Page: {PAGES[page_idx][0]}", (sw // 2 - 60, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 104, 148), 1)
        cv2.putText(canvas, f"Actions: {len(selection_log)}", (sw - 160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 104, 148), 1)

        # Find top intent for visual priority
        top_zone = max(zones, key=lambda z: z.posterior) if zones else None

        # Zones
        for z in zones:
            is_top = (top_zone and z.id == top_zone.id and z.posterior > 0.2)
            draw_zone(canvas, z, now, is_top)

        # Gaze dot
        cv2.circle(canvas, (gaze_x, gaze_y), 4, (0, 180, 255), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 6, (255, 255, 255), 1)

        # Footer
        eeg_label = f"EEG: {engagement:.2f}" if USE_EEG_local else "EEG: off"
        cv2.putText(canvas, eeg_label, (20, sh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 90), 1)

        if now < cooldown_until:
            cv2.putText(canvas, f"COOLDOWN {cooldown_until - now:.1f}s",
                        (sw // 2 - 50, sh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 120), 1)

        # Legend
        legend_x = sw - 250
        cv2.putText(canvas, "Zone types:", (legend_x, sh - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 100), 1)
        for j, (zt, cfg) in enumerate(ZONE_CONFIG.items()):
            if zt == ZoneType.PASSIVE:
                continue
            cv2.putText(canvas, f"{zt.value}: dwell={cfg['dwell_to_select']:.1f}s",
                        (legend_x + (j % 2) * 120, sh - 35 + (j // 2) * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, cfg["color"], 1)

        cv2.putText(canvas, "TAB=page  R=recal  ESC=quit",
                    (sw - 250, sh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 70), 1)

        cv2.imshow("Intent Engine", canvas)
        key = cv2.waitKey(16) & 0xFF

        if key == 27:
            break
        elif key == 9:  # TAB
            page_idx = (page_idx + 1) % len(PAGES)
            zones = PAGES[page_idx][1](sw, sh)
            engine = IntentEngine(zones)
            cooldown_until = 0
            print(f"  Page: {PAGES[page_idx][0]} ({len(zones)} zones)")
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(gaze, rows=5, cols=5, order="serpentine",
                                        pulse_d=1.0, cd_d=1.0, camera_index=cam_idx)
            filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
            filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Intent Engine", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Intent Engine", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    if eeg:
        eeg.stop()
    cv2.destroyAllWindows()

    print(f"\n  Session summary: {len(selection_log)} actions")
    for t, label, zt in selection_log:
        print(f"    [{zt:>7s}] {label}")
    print("\nDone.")


if __name__ == "__main__":
    main()
