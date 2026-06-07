#!/usr/bin/env python3
"""Demo server: eyetrax gaze + EEG → WebSocket → Netflix UI in browser.

All gaze tracking, attention engine, and EEG processing runs in Python.
The browser is just a display — receives element states via WebSocket.

Architecture:
  eyetrax (camera) → One Euro Filter → AttentionEngine → WebSocket → browser
  muse_bridge (EEG) → engagement computation ────────────┘

Usage:
    python3 server.py              # with EEG
    python3 server.py --no-eeg     # gaze only

Then open http://localhost:8090 in your browser.
"""

import asyncio
import json
import math
import os
import sys
import time
import threading
import numpy as np
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import websockets

from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from screeninfo import get_monitors
import cv2

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
# ATTENTION ENGINE (same as gaze-select)
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 80
HYSTERESIS_PAD = 15
EXIT_DELAY = 0.15

class ElementState:
    def __init__(self, id, x, y, w, h):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = 0.0
        self.dwell_time = 0.0
        self.gaze_in = False
        self.last_enter = 0.0
        self.last_exit = 0.0
        self.stability = 0.0

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2


class AttentionEngine:
    def __init__(self, n_elements):
        self.dwell_weight = 0.35
        self.stability_weight = 0.30
        self.proximity_weight = 0.15
        self.eeg_weight = 0.20
        self.dwell_saturate = 1.8 * (1 + max(0, math.log2(max(1, n_elements) / 4)))
        self.select_threshold = 0.75
        self.decay_rate = 0.91
        self.gaze_history = deque(maxlen=20)

    def update(self, elements, gaze_x, gaze_y, engagement):
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

        selected_id = None

        for el in elements.values():
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

    def reset_all(self, elements):
        for el in elements.values():
            el.confidence = 0
            el.gaze_in = False
            el.dwell_time = 0


# ═════════════════════════════════════════════════════════════
# EEG
# ═════════════════════════════════════════════════════════════

EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

def compute_engagement(eeg_window):
    if eeg_window is None or eeg_window.shape[1] < EEG_SR:
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
# MAIN — gaze loop + WebSocket server
# ═════════════════════════════════════════════════════════════

# Shared state between gaze thread and WebSocket
state = {
    "gaze_x": 0, "gaze_y": 0,
    "engagement": 0.5,
    "elements": {},  # id → ElementState
    "engine": None,
    "selected": None,
    "cooldown_until": 0,
    "ready": False,
}
ws_clients = set()


def gaze_thread_fn(cam_idx, gaze_estimator, eeg):
    """Runs gaze tracking + EEG in a background thread at ~30Hz."""
    filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.02)
    filter_y = OneEuroFilter(min_cutoff=0.4, beta=0.01)
    eng_history = deque(maxlen=15)

    cap = cv2.VideoCapture(cam_idx)

    print("[GAZE] Tracking started", flush=True)

    while state["ready"]:
        # EEG
        if eeg:
            eeg.pull()
            w = eeg.get_window(2.0)
            eng = compute_engagement(w)
            eng_history.append(eng)
            state["engagement"] = float(np.mean(eng_history))

        # Gaze
        ret, frame = cap.read()
        if ret:
            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                raw = gaze_estimator.predict(np.array([features]))[0]
                t = time.time()
                state["gaze_x"] = int(filter_x(raw[0], t))
                state["gaze_y"] = int(filter_y(raw[1], t))

        # Attention engine
        now = time.time()
        engine = state["engine"]
        elements = state["elements"]

        if engine and elements and now > state["cooldown_until"]:
            sel = engine.update(elements, state["gaze_x"], state["gaze_y"],
                                state["engagement"])
            if sel is not None:
                state["selected"] = sel
                print(f"[GAZE] SELECTED: {sel}", flush=True)
                engine.reset_all(elements)
                state["cooldown_until"] = now + 2.0

        time.sleep(0.030)

    cap.release()
    print("[GAZE] Stopped", flush=True)


async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)})", flush=True)

    try:
        async for msg in websocket:
            data = json.loads(msg)

            if data.get("type") == "register_elements":
                # Browser tells us where its elements are on screen
                elements = {}
                for el in data["elements"]:
                    elements[el["id"]] = ElementState(
                        el["id"], el["x"], el["y"], el["w"], el["h"]
                    )
                state["elements"] = elements
                state["engine"] = AttentionEngine(n_elements=len(elements))
                print(f"[WS] Registered {len(elements)} elements", flush=True)

            elif data.get("type") == "clear_selection":
                state["selected"] = None

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


async def broadcast_loop():
    """Send state to all connected browsers at ~20Hz."""
    while state["ready"]:
        if ws_clients:
            elements_data = {}
            for eid, el in state["elements"].items():
                elements_data[eid] = {
                    "confidence": round(el.confidence, 3),
                    "gaze_in": el.gaze_in,
                    "dwell_time": round(el.dwell_time, 2),
                }

            msg = json.dumps({
                "type": "state",
                "gaze_x": state["gaze_x"],
                "gaze_y": state["gaze_y"],
                "engagement": round(state["engagement"], 3),
                "elements": elements_data,
                "selected": state["selected"],
            })

            # Clear selection after sending once
            if state["selected"]:
                state["selected"] = None

            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            ws_clients.difference_update(dead)

        await asyncio.sleep(0.05)


def start_http_server():
    """Serve the HTML/CSS/JS from this directory."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("127.0.0.1", 8090), handler)
    print("[HTTP] Serving at http://localhost:8090", flush=True)
    httpd.serve_forever()


async def main():
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1470, 956
    print(f"Screen: {sw}x{sh}")

    # EEG
    eeg = None
    if USE_EEG:
        sys_path_backend = os.path.join(os.path.dirname(__file__), "..", "backend")
        if is_bridge_running():
            eeg = EEGBridgeClient()
            eeg.start()
            for _ in range(40):
                eeg.pull()
                time.sleep(0.05)
            print("EEG connected", flush=True)
        else:
            print("WARNING: No Muse bridge, running without EEG", flush=True)

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

    # Start HTTP server in background
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Start gaze thread
    state["ready"] = True
    gaze_t = threading.Thread(target=gaze_thread_fn, args=(cam_idx, gaze, eeg), daemon=True)
    gaze_t.start()

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "127.0.0.1", 8091)
    print("[WS] WebSocket on ws://127.0.0.1:8091", flush=True)
    print("\n  Open http://localhost:8090 in your browser\n", flush=True)

    # Broadcast loop
    try:
        await broadcast_loop()
    except KeyboardInterrupt:
        pass

    state["ready"] = False
    ws_server.close()
    if eeg:
        eeg.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
