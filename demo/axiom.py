#!/usr/bin/env python3
"""AXIOM — Brain-controlled web interaction.

Simple, effective architecture:
  - Webcam → eyetrax features (continuous, 30Hz)
  - Browser-based calibration (dots on the actual Netflix layout)
  - Nearest-element highlight with sticky pinning
  - EEG engagement boosts selection confidence
  - LinUCB bandit learns from neural reward

No OpenCV UI. No coordinate mismatch. Calibration IS the Netflix page.

Usage:
    python3 axiom.py              # full system
    python3 axiom.py --no-eeg     # gaze only
"""

import asyncio
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import websockets
import cv2

from eyetrax import GazeEstimator
from screeninfo import get_monitors

# ── EEG (optional) ──
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

# ── Weave (optional) ──
try:
    import weave
    weave.init("axiom-bci")
    print("[WEAVE] Connected", flush=True)
except Exception:
    pass

# ── OpenAI (optional) ──
try:
    from openai import OpenAI
    oai_client = OpenAI()
except Exception:
    oai_client = None


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
# EEG
# ═════════════════════════════════════════════════════════════

EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

def compute_engagement(eeg_window):
    if eeg_window is None or eeg_window.shape[1] < EEG_SR:
        return None
    def bp(data):
        out = data.copy()
        DataFilter.detrend(out, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
        DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
        nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
        psd = DataFilter.get_psd_welch(out, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
        return {n: float(DataFilter.get_band_power(psd, lo, hi)) for n, lo, hi in BANDS}
    bp1 = bp(eeg_window[1])
    bp2 = bp(eeg_window[2])
    alpha = (bp1["alpha"] + bp2["alpha"]) / 2
    theta = (bp1["theta"] + bp2["theta"]) / 2
    beta = (bp1["beta"] + bp2["beta"]) / 2
    return float(beta / (alpha + theta + 0.001))


# ═════════════════════════════════════════════════════════════
# ATTENTION ENGINE — simple, sticky, element-based
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 50
EXIT_DELAY = 0.4
SWITCH_DIST = 60

class ElementState:
    def __init__(self, id, x, y, w, h):
        self.id = id
        self.x = x; self.y = y; self.w = w; self.h = h
        self.confidence = 0.0
        self.dwell_time = 0.0
        self.pinned = False
        self.last_enter = 0.0

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


class AttentionEngine:
    def __init__(self, has_eeg=False):
        self.select_threshold = 0.75
        self.dwell_saturate = 2.0
        self.decay_rate = 0.94
        self.pinned_id = None
        self.empty_since = 0.0
        self.has_eeg = has_eeg

    def update(self, elements, gaze_x, gaze_y, engagement):
        now = time.time()

        # Find nearest element within range
        nearest_id = None
        nearest_dist = float('inf')
        for el in elements.values():
            if (el.x - MOTOR_PAD <= gaze_x <= el.x + el.w + MOTOR_PAD and
                    el.y - MOTOR_PAD <= gaze_y <= el.y + el.h + MOTOR_PAD):
                d = math.hypot(gaze_x - el.cx, gaze_y - el.cy)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_id = el.id

        # Sticky pin logic
        if nearest_id is not None:
            self.empty_since = 0
            if nearest_id != self.pinned_id:
                # Check if gaze moved far enough from current pin
                should_switch = True
                if self.pinned_id and self.pinned_id in elements:
                    p = elements[self.pinned_id]
                    if math.hypot(gaze_x - p.cx, gaze_y - p.cy) < SWITCH_DIST:
                        should_switch = False

                if should_switch:
                    # Unpin old
                    if self.pinned_id and self.pinned_id in elements:
                        old = elements[self.pinned_id]
                        old.pinned = False
                        old.dwell_time = 0
                        old.confidence *= 0.3
                    # Pin new
                    self.pinned_id = nearest_id
                    el = elements[nearest_id]
                    el.pinned = True
                    el.last_enter = now
                    el.dwell_time = 0
        else:
            if self.empty_since == 0:
                self.empty_since = now
            elif now - self.empty_since > EXIT_DELAY:
                if self.pinned_id and self.pinned_id in elements:
                    elements[self.pinned_id].pinned = False
                    elements[self.pinned_id].dwell_time = 0
                self.pinned_id = None

        # Update confidence
        selected_id = None
        for el in elements.values():
            if el.id == self.pinned_id:
                el.pinned = True
                el.dwell_time = now - el.last_enter
                dwell_score = min(1.0, el.dwell_time / self.dwell_saturate)

                if self.has_eeg and engagement is not None:
                    eeg_score = max(0, min(1, engagement))
                    raw = 0.55 * dwell_score + 0.45 * eeg_score
                else:
                    raw = dwell_score

                el.confidence = el.confidence * 0.6 + raw * 0.4
            else:
                el.pinned = False
                el.confidence *= self.decay_rate

            el.confidence = max(0, min(1, el.confidence))
            if el.confidence >= self.select_threshold:
                selected_id = el.id

        return selected_id

    def reset_all(self, elements):
        for el in elements.values():
            el.confidence = 0
            el.pinned = False
            el.dwell_time = 0
        self.pinned_id = None


# ═════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════

state = {
    "gaze_x": 0, "gaze_y": 0,
    "engagement": None,
    "elements": {},
    "engine": None,
    "selected": None,
    "cooldown_until": 0,
    "action_count": 0,
    "calibrated": False,
    "cal_features": [],  # collected during browser calibration
    "cal_targets": [],
    "ready": False,
}
ws_clients = set()


# ═════════════════════════════════════════════════════════════
# PERCEPTION LOOP — webcam + EEG, no UI
# ═════════════════════════════════════════════════════════════

def perception_loop(cam_idx, gaze_estimator, eeg):
    filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.008)
    filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.008)
    eng_history = deque(maxlen=15)
    tick = 0

    cap = cv2.VideoCapture(cam_idx)
    print("[PERCEPTION] Started", flush=True)

    while state["ready"]:
        tick += 1

        # Gaze — every frame
        ret, frame = cap.read()
        if ret:
            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                # Store latest features for calibration
                state["_latest_features"] = features

                if state["calibrated"]:
                    raw = gaze_estimator.predict(np.array([features]))[0]
                    t = time.time()
                    state["gaze_x"] = int(filter_x(raw[0], t))
                    state["gaze_y"] = int(filter_y(raw[1], t))

        # EEG — every 3rd frame
        if eeg and tick % 3 == 0:
            eeg.pull()
            w = eeg.get_window(2.0)
            eng = compute_engagement(w)
            if eng is not None:
                eng_history.append(eng)
                state["engagement"] = float(np.mean(eng_history))

        # Attention engine — every frame when calibrated
        if state["calibrated"]:
            now = time.time()
            engine = state["engine"]
            elements = state["elements"]

            if engine and elements and now > state["cooldown_until"]:
                sel = engine.update(elements, state["gaze_x"], state["gaze_y"],
                                    state["engagement"])
                if sel is not None:
                    state["selected"] = sel
                    state["action_count"] += 1
                    engine.reset_all(elements)
                    state["cooldown_until"] = now + 2.0
                    print(f"[SELECT] {sel} (eng={state['engagement']}, "
                          f"actions={state['action_count']})", flush=True)

        time.sleep(0.025)  # ~40Hz

    cap.release()


# ═════════════════════════════════════════════════════════════
# WEBSOCKET — calibration + state broadcast
# ═════════════════════════════════════════════════════════════

async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[WS] Client connected", flush=True)

    # Tell browser current calibration state
    await websocket.send(json.dumps({
        "type": "init",
        "calibrated": state["calibrated"],
        "has_eeg": state["engagement"] is not None,
    }))

    try:
        async for msg in websocket:
            data = json.loads(msg)

            if data["type"] == "cursor_train":
                # Continuous training: cursor position = ground truth for where eyes are
                screen_x = data["screen_x"]
                screen_y = data["screen_y"]
                features = state.get("_latest_features")
                if features is not None and state["calibrated"]:
                    state["cal_features"].append(features.copy())
                    state["cal_targets"].append([screen_x, screen_y])
                    # Retrain every 20 new points
                    if len(state["cal_features"]) % 20 == 0:
                        X = np.array(state["cal_features"])
                        y = np.array(state["cal_targets"])
                        state["_gaze_estimator"].train(X, y)
                        print(f"[CAL] Retrained on {len(state['cal_features'])} total points", flush=True)

            elif data["type"] == "cal_point":
                # Browser user clicked a calibration dot at these SCREEN coordinates
                screen_x = data["screen_x"]
                screen_y = data["screen_y"]
                features = state.get("_latest_features")

                if features is not None:
                    # Collect multiple frames for this point
                    state["cal_features"].append(features.copy())
                    state["cal_targets"].append([screen_x, screen_y])
                    n = len(state["cal_features"])
                    print(f"[CAL] Point {n}: ({screen_x}, {screen_y})", flush=True)

                    await websocket.send(json.dumps({
                        "type": "cal_ack", "count": n,
                    }))

            elif data["type"] == "cal_done":
                # Train the gaze model
                n = len(state["cal_features"])
                if n >= 5:
                    X = np.array(state["cal_features"])
                    y = np.array(state["cal_targets"])
                    state["_gaze_estimator"].train(X, y)
                    state["calibrated"] = True
                    print(f"[CAL] Trained on {n} points. Gaze active!", flush=True)

                    await websocket.send(json.dumps({
                        "type": "cal_complete", "n_points": n,
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "cal_error", "msg": f"Need at least 5 points, got {n}",
                    }))

            elif data["type"] == "register_elements":
                win_x = data.get("window_x", 0)
                win_y = data.get("window_y", 0)
                chrome_h = data.get("chrome_height", 0)

                elements = {}
                for el in data["elements"]:
                    elements[el["id"]] = ElementState(
                        el["id"],
                        el["x"] + win_x,
                        el["y"] + win_y + chrome_h,
                        el["w"], el["h"])

                state["elements"] = elements
                has_eeg = state["engagement"] is not None
                state["engine"] = AttentionEngine(has_eeg=has_eeg)
                print(f"[WS] {len(elements)} elements registered", flush=True)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


async def broadcast_loop():
    while state["ready"]:
        if ws_clients and state["calibrated"]:
            elements_data = {}
            for eid, el in state["elements"].items():
                elements_data[eid] = {
                    "confidence": round(el.confidence, 3),
                    "pinned": el.pinned,
                    "dwell_time": round(el.dwell_time, 2),
                }

            msg = json.dumps({
                "type": "state",
                "gaze_x": state["gaze_x"],
                "gaze_y": state["gaze_y"],
                "engagement": round(state["engagement"], 3) if state["engagement"] else None,
                "elements": elements_data,
                "selected": state["selected"],
                "action_count": state["action_count"],
                "threshold": round(state["engine"].select_threshold, 3) if state["engine"] else 0.75,
            })

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


def start_http():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    httpd = HTTPServer(("127.0.0.1", 8090), SimpleHTTPRequestHandler)
    print("[HTTP] http://localhost:8090", flush=True)
    httpd.serve_forever()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

async def main():
    print("AXIOM — Brain-Controlled Streaming")

    # Camera
    cam_idx = 1
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        ret, _ = cap.read()
        cap.release()
        if ret:
            cam_idx = idx
            break
    print(f"Camera: {cam_idx}")

    # Gaze estimator (NO calibration here — browser does it)
    gaze = GazeEstimator(model_name="tiny_mlp")
    state["_gaze_estimator"] = gaze

    # EEG
    eeg = None
    if USE_EEG:
        if is_bridge_running():
            eeg = EEGBridgeClient()
            eeg.start()
            for _ in range(40):
                eeg.pull()
                time.sleep(0.05)
            print("EEG: connected", flush=True)
        else:
            print("EEG: no bridge, running without", flush=True)

    # Start servers
    threading.Thread(target=start_http, daemon=True).start()

    state["ready"] = True
    threading.Thread(target=perception_loop, args=(cam_idx, gaze, eeg), daemon=True).start()

    ws_server = await websockets.serve(ws_handler, "127.0.0.1", 8091)
    print(f"\n  Open http://localhost:8090\n  Calibrate by clicking the dots.\n", flush=True)

    try:
        await broadcast_loop()
    except KeyboardInterrupt:
        pass

    state["ready"] = False
    ws_server.close()
    if eeg:
        eeg.stop()


if __name__ == "__main__":
    asyncio.run(main())
