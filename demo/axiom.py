#!/usr/bin/env python3
"""AXIOM — Brain-controlled web interaction via multi-agent orchestration.

Architecture:
  Tier 1 (pure math, 20Hz):
    PerceptionAgent: EEG → engagement/focus/errp, Gaze → screen position
    IntentAgent: fuse signals → per-element confidence → threshold check

  Tier 2 (LLM-powered, async):
    PageAgent: OpenAI → classify DOM into semantic zones (on page load)
    SelfImprovementAgent: Weave feedback → threshold tuning (every 30s)

  Core RL loop:
    LinUCB contextual bandit with neural reward:
      r = 0.7 × errp_signal + 0.3 × engagement_delta

  NeuroLLM:
    Brain state → LLM temperature + prompt persona injection

Usage:
    python3 axiom.py              # full system
    python3 axiom.py --no-eeg     # gaze only, simulated engagement
"""

import asyncio
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import redis
import weave
import websockets

# ── Gaze ──
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from screeninfo import get_monitors
import cv2

# ── EEG ──
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

# ── OpenAI ──
from openai import OpenAI
oai_client = OpenAI()  # reads OPENAI_API_KEY from env


# ═════════════════════════════════════════════════════════════
# WEAVE INIT
# ═════════════════════════════════════════════════════════════

try:
    weave.init("axiom-bci")
    HAS_WEAVE = True
    print("[WEAVE] Connected", flush=True)
except Exception as e:
    HAS_WEAVE = False
    print(f"[WEAVE] Not available ({e}), continuing without tracing", flush=True)
    # Make weave.op a no-op decorator
    _real_weave_op = weave.op
    weave.op = lambda f=None, **kw: f if f else (lambda fn: fn)


# ═════════════════════════════════════════════════════════════
# REDIS — real-time nervous system
# ═════════════════════════════════════════════════════════════

try:
    rdb = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    rdb.ping()
    HAS_REDIS = True
    print("[REDIS] Connected", flush=True)
except Exception:
    rdb = None
    HAS_REDIS = False
    print("[REDIS] Not available, continuing without", flush=True)

SESSION_ID = f"session:{int(time.time())}"


class RedisBrainBus:
    """Streams brain state to Redis for multi-agent consumption.
    Tracks rolling engagement baseline for drift detection."""

    def __init__(self):
        if not HAS_REDIS:
            return
        # Stream for real-time brain state fan-out
        self.stream_key = f"stream:brain:{SESSION_ID}"
        # Rolling engagement baseline (5-minute buckets)
        self.baseline_key = f"baseline:engagement:{SESSION_ID}"
        self.baseline_values = []
        self.baseline_window = 300  # 5 minutes
        self.last_baseline_update = time.time()
        # Session memory
        self.session_key = f"session:memory:{SESSION_ID}"
        rdb.hset(self.session_key, mapping={
            "start_time": str(time.time()),
            "action_count": "0",
            "selections": "[]",
            "initial_threshold": "0.75",
        })
        rdb.expire(self.session_key, 7200)

    def publish_brain_state(self, brain_state, gaze_x, gaze_y, engagement):
        if not HAS_REDIS:
            return
        try:
            rdb.xadd(self.stream_key, {
                "ts": str(int(time.time() * 1000)),
                "engagement": f"{engagement:.3f}",
                "focus": f"{brain_state['focus']:.3f}",
                "theta_beta": f"{brain_state['theta_beta']:.3f}",
                "alpha": f"{brain_state['alpha']:.3f}",
                "beta": f"{brain_state['beta']:.3f}",
                "gaze_x": str(gaze_x),
                "gaze_y": str(gaze_y),
            }, maxlen=6000, approximate=True)  # ~5 min at 20Hz

            # Track rolling baseline for drift detection
            self.baseline_values.append(engagement)
            now = time.time()
            if now - self.last_baseline_update > 30:  # every 30s
                self._update_baseline()
                self.last_baseline_update = now

        except Exception:
            pass

    def _update_baseline(self):
        if not self.baseline_values:
            return
        mean = float(np.mean(self.baseline_values[-600:]))  # last 30s worth
        std = float(np.std(self.baseline_values[-600:]))
        try:
            rdb.hset(self.baseline_key, mapping={
                "mean": f"{mean:.4f}",
                "std": f"{std:.4f}",
                "n_samples": str(len(self.baseline_values)),
                "updated_at": str(time.time()),
            })
            rdb.expire(self.baseline_key, 7200)
        except Exception:
            pass

    def detect_drift(self, current_engagement):
        """Check if current engagement has drifted from baseline."""
        if not HAS_REDIS:
            return None
        try:
            baseline = rdb.hgetall(self.baseline_key)
            if not baseline or "mean" not in baseline:
                return None
            b_mean = float(baseline["mean"])
            b_std = float(baseline["std"])
            if b_std < 0.01:
                return None
            z_score = (current_engagement - b_mean) / b_std
            if abs(z_score) > 2.0:
                return {"drift": True, "z_score": z_score, "baseline_mean": b_mean,
                        "current": current_engagement, "direction": "up" if z_score > 0 else "down"}
            return None
        except Exception:
            return None

    def record_selection(self, element_id, engagement, brain_persona, reward):
        if not HAS_REDIS:
            return
        try:
            rdb.hincrby(self.session_key, "action_count", 1)
            # Append to selection history
            history = json.loads(rdb.hget(self.session_key, "selections") or "[]")
            history.append({
                "element": element_id,
                "engagement": round(engagement, 3),
                "persona": brain_persona,
                "reward": round(reward, 3),
                "time": time.time(),
            })
            # Keep last 100
            rdb.hset(self.session_key, "selections", json.dumps(history[-100:]))
        except Exception:
            pass

    def get_session_summary(self):
        if not HAS_REDIS:
            return {}
        try:
            data = rdb.hgetall(self.session_key)
            return data
        except Exception:
            return {}


brain_bus = RedisBrainBus()


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
# EEG PROCESSING
# ═════════════════════════════════════════════════════════════

EEG_SR = 256
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

def compute_band_powers(data_1ch):
    if len(data_1ch) < EEG_SR:
        return {n: 0.0 for n, _, _ in BANDS}
    out = data_1ch.copy()
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4,
                                FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
    nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
    psd = DataFilter.get_psd_welch(out, nfft, nfft // 2, EEG_SR,
                                    WindowOperations.HANNING.value)
    return {n: float(DataFilter.get_band_power(psd, lo, hi)) for n, lo, hi in BANDS}


@weave.op
def compute_brain_state(eeg_window):
    """Extract engagement, focus, and raw band powers from EEG."""
    if eeg_window is None or eeg_window.shape[1] < EEG_SR:
        return {"engagement": 0.5, "focus": 0.5, "theta_beta": 1.0,
                "alpha": 0.0, "beta": 0.0, "theta": 0.0}

    bp1 = compute_band_powers(eeg_window[1])  # AF7
    bp2 = compute_band_powers(eeg_window[2])  # AF8

    alpha = (bp1["alpha"] + bp2["alpha"]) / 2
    theta = (bp1["theta"] + bp2["theta"]) / 2
    beta = (bp1["beta"] + bp2["beta"]) / 2

    engagement = beta / (alpha + theta + 0.001)
    theta_beta = theta / (beta + 0.001)

    return {
        "engagement": float(engagement),
        "focus": float(1.0 / (1.0 + theta_beta)),
        "theta_beta": float(theta_beta),
        "alpha": float(alpha),
        "beta": float(beta),
        "theta": float(theta),
    }


def detect_errp(eeg_window, action_time):
    """Simple ErrP detection: check for frontal negativity 250-500ms post-action."""
    if eeg_window is None or eeg_window.shape[1] < EEG_SR:
        return 0.0

    elapsed = time.time() - action_time
    if elapsed < 0.2 or elapsed > 0.8:
        return 0.0

    # Look at frontal channels (AF7=1, AF8=2) in the 250-500ms window
    start_sample = max(0, eeg_window.shape[1] - int(0.5 * EEG_SR))
    end_sample = max(0, eeg_window.shape[1] - int(0.2 * EEG_SR))
    if end_sample <= start_sample:
        return 0.0

    segment = eeg_window[1:3, start_sample:end_sample]  # AF7, AF8
    if segment.shape[1] < 10:
        return 0.0

    # ErrP signature: negative deflection followed by positive
    avg = segment.mean(axis=0)
    mid = len(avg) // 2
    first_half = avg[:mid].mean()
    second_half = avg[mid:].mean()

    # Negative-then-positive pattern
    if first_half < -20 and second_half > 10:
        return 0.8
    elif first_half < -10:
        return 0.4
    return 0.1


# ═════════════════════════════════════════════════════════════
# LinUCB CONTEXTUAL BANDIT
# ═════════════════════════════════════════════════════════════

class LinUCBBandit:
    """LinUCB contextual bandit with neural reward signal."""

    def __init__(self, n_actions, context_dim=12, alpha=1.5):
        self.n_actions = n_actions
        self.d = context_dim
        self.alpha = alpha
        # Per-arm parameters
        self.A = [np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]
        self.total_updates = 0
        self.reward_history = []

    def select_action(self, context):
        """Select action using UCB. Returns (action_idx, confidence)."""
        ctx = np.array(context, dtype=np.float64)[:self.d]
        if len(ctx) < self.d:
            ctx = np.concatenate([ctx, np.zeros(self.d - len(ctx))])

        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            pred = ctx @ theta
            ucb = self.alpha * np.sqrt(ctx @ A_inv @ ctx)
            scores[a] = pred + ucb

        best = int(np.argmax(scores))
        # Confidence via softmax
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        return best, float(probs[best])

    def update(self, context, action, reward):
        """Update arm with observed reward."""
        ctx = np.array(context, dtype=np.float64)[:self.d]
        if len(ctx) < self.d:
            ctx = np.concatenate([ctx, np.zeros(self.d - len(ctx))])

        self.A[action] += np.outer(ctx, ctx)
        self.b[action] += reward * ctx
        self.total_updates += 1
        self.reward_history.append(reward)

    @property
    def avg_reward(self):
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history[-50:])


# ═════════════════════════════════════════════════════════════
# ATTENTION ENGINE (proven from gaze-select)
# ═════════════════════════════════════════════════════════════

MOTOR_PAD = 90
EXIT_DELAY = 0.4  # sticky — don't unhighlight for 400ms

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
        self.highlighted = False
        self.last_enter = 0.0
        self.last_exit = 0.0
        self.stability = 0.0

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


class AttentionEngine:
    """Sticky, element-based attention. No cursor — just highlights.

    Philosophy: the user is GUIDING us, not hiding. If gaze is near
    an element, commit to highlighting it. Only switch when gaze is
    clearly on a DIFFERENT element. Looking away = rejection signal.
    """

    def __init__(self, n_elements):
        self.dwell_weight = 0.30
        self.stability_weight = 0.20
        self.proximity_weight = 0.15
        self.eeg_weight = 0.35
        self.dwell_saturate = 1.8 * (1 + max(0, math.log2(max(1, n_elements) / 4)))
        self.select_threshold = 0.75
        self.decay_rate = 0.94  # slower decay — stickier
        self.gaze_history = deque(maxlen=20)
        self.pinned_id = None  # currently highlighted element
        self.pinned_since = 0.0
        self.empty_gaze_since = 0.0  # when gaze last left all elements

    def _find_nearest(self, elements, gaze_x, gaze_y):
        """Find the element closest to gaze, within motor range."""
        best_id = None
        best_dist = float('inf')
        for el in elements.values():
            if (el.x - MOTOR_PAD <= gaze_x <= el.x + el.w + MOTOR_PAD and
                    el.y - MOTOR_PAD <= gaze_y <= el.y + el.h + MOTOR_PAD):
                dist = math.hypot(gaze_x - el.cx, gaze_y - el.cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = el.id
        return best_id

    def update(self, elements, gaze_x, gaze_y, engagement):
        now = time.time()
        self.gaze_history.append((gaze_x, gaze_y, now))

        # Gaze velocity → stability
        gaze_speed = 0.0
        if len(self.gaze_history) >= 3:
            pts = list(self.gaze_history)
            dists = [math.hypot(pts[k][0] - pts[k-1][0], pts[k][1] - pts[k-1][1])
                     for k in range(1, len(pts))]
            span = pts[-1][2] - pts[0][2]
            if span > 0:
                gaze_speed = sum(dists) / span
        stability = max(0, min(1, 1.0 - (gaze_speed - 40) / 400))

        # Find which element gaze is nearest to
        nearest_id = self._find_nearest(elements, gaze_x, gaze_y)

        # Sticky pin logic:
        # - If gaze is on a NEW element, switch pin immediately (user is guiding)
        # - If gaze is on the SAME element, keep building confidence
        # - If gaze is on NO element, keep current pin for EXIT_DELAY, then unpin
        if nearest_id is not None:
            self.empty_gaze_since = 0
            if nearest_id != self.pinned_id:
                # User moved to a different element — switch immediately
                if self.pinned_id and self.pinned_id in elements:
                    elements[self.pinned_id].gaze_in = False
                    elements[self.pinned_id].dwell_time = 0
                self.pinned_id = nearest_id
                self.pinned_since = now
                el = elements[nearest_id]
                el.gaze_in = True
                el.last_enter = now
                el.dwell_time = 0
                el.highlighted = True
        else:
            # Gaze is in empty space
            if self.empty_gaze_since == 0:
                self.empty_gaze_since = now
            elif now - self.empty_gaze_since > EXIT_DELAY:
                # Gaze has been away long enough — unpin
                if self.pinned_id and self.pinned_id in elements:
                    elements[self.pinned_id].gaze_in = False
                    elements[self.pinned_id].dwell_time = 0
                    elements[self.pinned_id].highlighted = False
                self.pinned_id = None

        # Update all elements
        selected_id = None
        for el in elements.values():
            is_pinned = (el.id == self.pinned_id)

            if is_pinned:
                el.gaze_in = True
                el.dwell_time = now - el.last_enter
                el.stability = stability
                el.highlighted = True

                # Compute confidence
                dist = math.hypot(gaze_x - el.cx, gaze_y - el.cy)
                max_dist = math.hypot(el.w, el.h) / 2 + MOTOR_PAD
                proximity = max(0, 1.0 - dist / max_dist) if max_dist > 0 else 0

                dwell_score = min(1.0, el.dwell_time / self.dwell_saturate)
                eeg_score = max(0, min(1, engagement))

                raw = (self.dwell_weight * dwell_score +
                       self.stability_weight * stability +
                       self.proximity_weight * proximity +
                       self.eeg_weight * eeg_score)
                el.confidence = el.confidence * 0.65 + raw * 0.35
            else:
                el.gaze_in = False
                el.highlighted = False
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
            el.highlighted = False
        self.pinned_id = None
        self.empty_gaze_since = 0


# ═════════════════════════════════════════════════════════════
# NEUROLLM — Brain-state-conditioned LLM
# ═════════════════════════════════════════════════════════════

BRAIN_PERSONAS = {
    "focused": "The user is highly focused and engaged. They know what they want. "
               "Be precise and direct. Confirm their current trajectory.",
    "browsing": "The user is casually browsing. They're exploring, not deciding. "
                "Present options broadly. Don't rush them toward action.",
    "deciding": "The user is actively evaluating options. They're comparing and deliberating. "
                "Highlight key differences. Present clear comparisons.",
    "confused": "The user seems uncertain or overwhelmed. Their cognitive load is high. "
                "Simplify choices. Reduce visual clutter. Explain clearly.",
}

def get_brain_persona(brain_state):
    """Map brain metrics to a persona for LLM prompt injection."""
    eng = brain_state["engagement"]
    tb = brain_state["theta_beta"]
    focus = brain_state["focus"]

    if eng > 0.7 and focus > 0.6:
        return "focused", BRAIN_PERSONAS["focused"]
    elif tb > 2.0:
        return "confused", BRAIN_PERSONAS["confused"]
    elif eng > 0.4 and eng < 0.7:
        return "deciding", BRAIN_PERSONAS["deciding"]
    else:
        return "browsing", BRAIN_PERSONAS["browsing"]


def neural_temperature(engagement):
    """Map engagement to LLM temperature. High focus → low temp → precise."""
    return max(0.1, min(1.0, 1.0 - engagement * 0.8))


@weave.op
def page_classify(page_title, element_labels):
    """Use OpenAI to classify page elements into semantic zones."""
    persona_name, persona = get_brain_persona(shared_state["brain_state"])
    temp = neural_temperature(shared_state["brain_state"]["engagement"])

    try:
        response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temp,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"""You classify web UI elements for a brain-computer interface.
{persona}
Classify each element as: NAV, ACTION, CONTENT, LINK, or PASSIVE.
Predict the top 3 likely-next-click elements.
Return JSON: {{"elements": [{{"id": "...", "type": "...", "priority": 0.0-1.0}}], "likely_next": ["id1","id2","id3"]}}"""},
                {"role": "user", "content": f"Page: {page_title}\nElements: {json.dumps(element_labels)}"}
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[PAGE] Classification error: {e}", flush=True)
        return {"elements": [], "likely_next": []}


# ═════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT AGENT
# ═════════════════════════════════════════════════════════════

class SelfImprovementAgent:
    """Adjusts thresholds and bandit parameters based on accumulated feedback."""

    def __init__(self):
        self.action_log = []  # (time, element_id, engagement, reward)
        self.last_improvement = time.time()
        self.improvement_interval = 30.0  # seconds
        self.threshold_history = []

    @weave.op
    def record_action(self, element_id, engagement, brain_state, reward):
        self.action_log.append({
            "time": time.time(),
            "element_id": element_id,
            "engagement": engagement,
            "focus": brain_state["focus"],
            "theta_beta": brain_state["theta_beta"],
            "reward": reward,
        })
        return {"logged": True, "total_actions": len(self.action_log)}

    @weave.op
    def maybe_improve(self, engine, bandit):
        """Check if it's time to adjust thresholds."""
        now = time.time()
        if now - self.last_improvement < self.improvement_interval:
            return None
        if len(self.action_log) < 5:
            return None

        self.last_improvement = now
        recent = self.action_log[-20:]

        # Compute metrics
        avg_reward = np.mean([a["reward"] for a in recent])
        pos_rate = np.mean([1 if a["reward"] > 0 else 0 for a in recent])
        avg_engagement = np.mean([a["engagement"] for a in recent])

        adjustments = {}

        # If too many false positives (low reward), raise threshold
        if avg_reward < 0.3 and pos_rate < 0.6:
            engine.select_threshold = min(0.90, engine.select_threshold + 0.03)
            adjustments["select_threshold"] = engine.select_threshold
            adjustments["reason"] = "too many false positives"

        # If accuracy is good but slow, lower threshold slightly
        elif pos_rate > 0.8 and avg_reward > 0.5:
            engine.select_threshold = max(0.55, engine.select_threshold - 0.02)
            adjustments["select_threshold"] = engine.select_threshold
            adjustments["reason"] = "accuracy good, can be faster"

        # Adjust EEG weight based on how predictive engagement is
        high_eng_actions = [a for a in recent if a["engagement"] > 0.6]
        low_eng_actions = [a for a in recent if a["engagement"] <= 0.6]
        if high_eng_actions and low_eng_actions:
            high_reward = np.mean([a["reward"] for a in high_eng_actions])
            low_reward = np.mean([a["reward"] for a in low_eng_actions])
            if high_reward > low_reward + 0.2:
                engine.eeg_weight = min(0.45, engine.eeg_weight + 0.02)
                adjustments["eeg_weight"] = engine.eeg_weight
                adjustments["eeg_reason"] = "engagement is predictive, increasing weight"

        if adjustments:
            self.threshold_history.append((now, adjustments))
            print(f"[IMPROVE] {adjustments}", flush=True)

        return adjustments if adjustments else None


# ═════════════════════════════════════════════════════════════
# SHARED STATE
# ═════════════════════════════════════════════════════════════

shared_state = {
    "gaze_x": 0, "gaze_y": 0,
    "engagement": 0.5,
    "brain_state": {"engagement": 0.5, "focus": 0.5, "theta_beta": 1.0,
                    "alpha": 0.0, "beta": 0.0, "theta": 0.0},
    "brain_persona": "browsing",
    "elements": {},
    "engine": None,
    "bandit": None,
    "improver": SelfImprovementAgent(),
    "selected": None,
    "cooldown_until": 0,
    "last_action_time": 0,
    "prev_engagement": 0.5,
    "action_count": 0,
    "bandit_avg_reward": 0.0,
    "ready": False,
}
ws_clients = set()


# ═════════════════════════════════════════════════════════════
# PERCEPTION + INTENT LOOP (Tier 1, 30Hz, no LLM)
# ═════════════════════════════════════════════════════════════

def perception_loop(cam_idx, gaze_estimator, eeg):
    filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.008)
    filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.008)
    eng_history = deque(maxlen=15)

    cap = cv2.VideoCapture(cam_idx)
    print("[PERCEPTION] Started", flush=True)

    while shared_state["ready"]:
        # EEG
        if eeg:
            eeg.pull()
            w = eeg.get_window(2.0)
            brain = compute_brain_state(w)
            shared_state["brain_state"] = brain
            eng_history.append(brain["engagement"])
            shared_state["engagement"] = float(np.mean(eng_history))

            persona_name, _ = get_brain_persona(brain)
            shared_state["brain_persona"] = persona_name

            # Publish to Redis for multi-agent consumption
            brain_bus.publish_brain_state(brain, shared_state["gaze_x"],
                                          shared_state["gaze_y"],
                                          shared_state["engagement"])

            # Check for calibration drift
            drift = brain_bus.detect_drift(shared_state["engagement"])
            if drift:
                print(f"[DRIFT] Engagement drifted {drift['direction']} "
                      f"(z={drift['z_score']:.1f}, baseline={drift['baseline_mean']:.3f}, "
                      f"current={drift['current']:.3f})", flush=True)

            # ErrP check after actions
            if shared_state["last_action_time"] > 0:
                errp = detect_errp(w, shared_state["last_action_time"])
                elapsed = time.time() - shared_state["last_action_time"]
                if elapsed > 0.8:
                    # Window expired — compute final reward
                    eng_delta = shared_state["engagement"] - shared_state["prev_engagement"]
                    reward = 0.7 * (1.0 - errp) + 0.3 * np.clip(eng_delta * 5, -1, 1)

                    bandit = shared_state["bandit"]
                    if bandit and hasattr(shared_state, "_last_context"):
                        bandit.update(shared_state["_last_context"],
                                      shared_state["_last_action"], reward)
                        shared_state["bandit_avg_reward"] = bandit.avg_reward

                    # Log to self-improvement agent
                    shared_state["improver"].record_action(
                        shared_state.get("_last_element_id", ""),
                        shared_state["engagement"],
                        brain, reward
                    )

                    # Record in Redis session memory
                    brain_bus.record_selection(
                        shared_state.get("_last_element_id", ""),
                        shared_state["engagement"],
                        shared_state["brain_persona"],
                        float(reward),
                    )

                    shared_state["last_action_time"] = 0

        # Gaze
        ret, frame = cap.read()
        if ret:
            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                raw = gaze_estimator.predict(np.array([features]))[0]
                t = time.time()
                shared_state["gaze_x"] = int(filter_x(raw[0], t))
                shared_state["gaze_y"] = int(filter_y(raw[1], t))

        # Intent (attention engine)
        now = time.time()
        engine = shared_state["engine"]
        elements = shared_state["elements"]

        if engine and elements and now > shared_state["cooldown_until"]:
            sel = engine.update(elements, shared_state["gaze_x"],
                                shared_state["gaze_y"], shared_state["engagement"])
            if sel is not None:
                shared_state["selected"] = sel
                shared_state["prev_engagement"] = shared_state["engagement"]
                shared_state["last_action_time"] = now
                shared_state["action_count"] += 1

                # Build context for bandit
                el = elements[sel]
                context = [
                    shared_state["gaze_x"] / 1470.0,
                    shared_state["gaze_y"] / 956.0,
                    el.dwell_time,
                    shared_state["brain_state"]["alpha"],
                    shared_state["brain_state"]["beta"],
                    shared_state["brain_state"]["theta"],
                    shared_state["engagement"],
                    shared_state["brain_state"]["focus"],
                    shared_state["brain_state"]["theta_beta"],
                    el.confidence,
                    el.stability,
                    shared_state["action_count"] / 100.0,
                ]
                shared_state["_last_context"] = context
                shared_state["_last_action"] = list(elements.keys()).index(sel)
                shared_state["_last_element_id"] = sel

                engine.reset_all(elements)
                shared_state["cooldown_until"] = now + 2.0

                print(f"[INTENT] Selected {sel} (engagement={shared_state['engagement']:.2f}, "
                      f"persona={shared_state['brain_persona']})", flush=True)

        # Self-improvement check
        if engine:
            shared_state["improver"].maybe_improve(engine, shared_state.get("bandit"))

        time.sleep(0.030)

    cap.release()
    print("[PERCEPTION] Stopped", flush=True)


# ═════════════════════════════════════════════════════════════
# WEBSOCKET SERVER
# ═════════════════════════════════════════════════════════════

async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"[WS] Client connected ({len(ws_clients)})", flush=True)
    try:
        async for msg in websocket:
            data = json.loads(msg)
            if data.get("type") == "register_elements":
                elements = {}
                for el in data["elements"]:
                    elements[el["id"]] = ElementState(
                        el["id"], el["x"], el["y"], el["w"], el["h"])
                shared_state["elements"] = elements
                shared_state["engine"] = AttentionEngine(n_elements=len(elements))
                shared_state["bandit"] = LinUCBBandit(n_actions=len(elements))
                print(f"[WS] Registered {len(elements)} elements, bandit initialized", flush=True)

                # Async page classification
                labels = [{"id": el["id"], "label": el.get("label", el["id"])}
                          for el in data["elements"]]
                title = data.get("page_title", "Unknown Page")
                threading.Thread(target=lambda: page_classify(title, labels),
                                 daemon=True).start()

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


async def broadcast_loop():
    while shared_state["ready"]:
        if ws_clients:
            elements_data = {}
            engine = shared_state["engine"]
            pinned = engine.pinned_id if engine else None
            for eid, el in shared_state["elements"].items():
                elements_data[eid] = {
                    "confidence": round(el.confidence, 3),
                    "gaze_in": el.gaze_in,
                    "highlighted": el.highlighted,
                    "pinned": eid == pinned,
                    "dwell_time": round(el.dwell_time, 2),
                }

            msg = json.dumps({
                "type": "state",
                "gaze_x": shared_state["gaze_x"],
                "gaze_y": shared_state["gaze_y"],
                "engagement": round(shared_state["engagement"], 3),
                "brain_persona": shared_state["brain_persona"],
                "brain_state": {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in shared_state["brain_state"].items()},
                "elements": elements_data,
                "selected": shared_state["selected"],
                "action_count": shared_state["action_count"],
                "bandit_avg_reward": round(shared_state["bandit_avg_reward"], 3),
                "threshold": round(shared_state["engine"].select_threshold, 3)
                             if shared_state["engine"] else 0.75,
                "session_id": SESSION_ID,
                "has_redis": HAS_REDIS,
            })

            if shared_state["selected"]:
                shared_state["selected"] = None

            dead = set()
            for ws in list(ws_clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            ws_clients.difference_update(dead)

        await asyncio.sleep(0.05)


def start_http_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("127.0.0.1", 8090), handler)
    print("[HTTP] http://localhost:8090", flush=True)
    httpd.serve_forever()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

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
        sys_path = os.path.join(os.path.dirname(__file__), "..", "backend")
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

    # Start HTTP server
    threading.Thread(target=start_http_server, daemon=True).start()

    # Start perception loop
    shared_state["ready"] = True
    threading.Thread(target=perception_loop, args=(cam_idx, gaze, eeg), daemon=True).start()

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "127.0.0.1", 8091)
    print("[WS] ws://127.0.0.1:8091", flush=True)
    print("\n  Open http://localhost:8090 in your browser\n", flush=True)

    try:
        await broadcast_loop()
    except KeyboardInterrupt:
        pass

    shared_state["ready"] = False
    ws_server.close()
    if eeg:
        eeg.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
