#!/usr/bin/env python3
"""Axiom v2 — Universal Brain-Computer Interface Orchestrator.

Orchestrates:
  1. Muse S EEG (BrainFlow) -> EEGNet features + HMM brain state at 2Hz
  2. Webcam gaze tracker (MediaPipe) -> screen position at 15Hz
  3. Playwright browser -> any-website control + element positions
  4. Neural pipeline -> Bayesian intent accumulation + UCB bandit action selection
  5. LLM (AWS Bedrock Claude) -> optional smart action decisions

Phases:
  - startup: connecting hardware + launching browser
  - gaze_cal: 9-point gaze calibration
  - calibrate: EEG calibration
  - live: system predicts and acts from brain + gaze

Progressive commitment flow:
  gaze fixation -> Bayesian intent accumulation -> commitment ring ->
  UCB bandit action selection -> execution -> ErrP monitoring -> bandit reward update
"""

import asyncio
import json
import sys
import time
import os
import numpy as np
import websockets
from dataclasses import asdict

from brain_state import BrainStateEngine, BrainState
from gaze_tracker import GazeTracker, GazeResult
from universal_controller import UniversalController, WebElement
from llm_engine import AxiomLLM

# Neural pipeline components — imported with fallback stubs
try:
    from neural_pipeline import (
        EEGNetExtractor,
        BrainStateHMM,
        ErrPDetector,
        BayesianIntentAccumulator,
        NeuralUCBBandit,
    )
    HAS_NEURAL_PIPELINE = True
except ImportError:
    HAS_NEURAL_PIPELINE = False

SIM_MODE = "--sim" in sys.argv
USE_LLM = "--llm" in sys.argv or os.environ.get("USE_LLM")
TEST_MODE = "--test" in sys.argv
EEG_SR = 256
GAZE_HZ = 15
BRAIN_HZ = 2
SCAN_HZ = 0.5
INTENT_HZ = 5

clients: set = set()

# ── HMM State Constants ──────────────────────────────────────────

HMM_STATES = ["browsing", "searching", "intending", "resting", "error"]
HMM_COLORS = {
    "browsing": "#00d4ff",
    "searching": "#ff6b9d",
    "intending": "#00ff88",
    "resting": "#888888",
    "error": "#ff4444",
}

# ── Supported Actions ────────────────────────────────────────────

SUPPORTED_ACTIONS = [
    "click", "scroll_down", "scroll_up", "back", "forward",
    "open_tab", "close_tab", "type_text", "none",
]


# ── Fixation Detector ────────────────────────────────────────────

class FixationDetector:
    def __init__(self, dispersion_px=60, min_duration_ms=200, max_history=30):
        self._disp_thresh = dispersion_px
        self._min_samples = max(2, int(min_duration_ms / 1000 * GAZE_HZ))
        self._history: list[tuple[float, float, float]] = []
        self._max = max_history
        self._fixation_start = 0.0
        self.fixating = False
        self.fixation_x = 0.0
        self.fixation_y = 0.0
        self.fixation_duration = 0.0

    def update(self, x: float, y: float, t: float):
        self._history.append((x, y, t))
        if len(self._history) > self._max:
            self._history.pop(0)

        if len(self._history) < self._min_samples:
            self.fixating = False
            return

        recent = self._history[-self._min_samples:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        disp = (max(xs) - min(xs)) + (max(ys) - min(ys))

        if disp < self._disp_thresh:
            if not self.fixating:
                self._fixation_start = recent[0][2]
            self.fixating = True
            self.fixation_x = sum(xs) / len(xs)
            self.fixation_y = sum(ys) / len(ys)
            self.fixation_duration = t - self._fixation_start
        else:
            self.fixating = False
            self.fixation_duration = 0.0


# ── EEG Source (real or simulated) ───────────────────────────────

class EEGSource:
    def __init__(self, sim=False):
        self._sim = sim
        self._board = None
        self._channels = None
        self._ring = np.zeros((4, EEG_SR * 10))
        self._pos = 0
        self._total = 0

    def start(self):
        if self._sim:
            print("[EEG] Simulation mode", flush=True)
            return
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        params = BrainFlowInputParams()
        params.timeout = 5
        board_id = BoardIds.MUSE_S_BOARD.value
        self._board = BoardShim(board_id, params)
        print("[EEG] Scanning for Muse S...", flush=True)
        self._board.prepare_session()
        self._board.start_stream(num_samples=450000)
        self._channels = BoardShim.get_eeg_channels(board_id)[:4]
        print("[EEG] Connected, streaming at 256Hz", flush=True)

    def stop(self):
        if self._board:
            self._board.stop_stream()
            self._board.release_session()

    def pull(self) -> np.ndarray:
        if self._sim:
            n = int(EEG_SR * 0.05)
            t = np.linspace(self._total / EEG_SR, (self._total + n) / EEG_SR, n)
            data = np.zeros((4, n))
            for ch in range(4):
                alpha = 15 * np.sin(2 * np.pi * 10 * t + ch)
                beta = 5 * np.sin(2 * np.pi * 20 * t + ch * 0.5)
                theta = 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                noise = np.random.randn(n) * 3
                data[ch] = alpha + beta + theta + noise
            self._total += n
            self._write_ring(data)
            return data

        raw = self._board.get_board_data(num_samples=128)
        if raw.shape[1] == 0:
            return np.zeros((4, 0))
        eeg = raw[self._channels, :]
        self._write_ring(eeg)
        return eeg

    def _write_ring(self, data):
        n = data.shape[1]
        cap = self._ring.shape[1]
        for i in range(n):
            self._ring[:, self._pos % cap] = data[:, i]
            self._pos += 1
        self._total = max(self._total, self._pos)

    def get_window(self, seconds: float) -> np.ndarray:
        n = int(seconds * EEG_SR)
        cap = self._ring.shape[1]
        n = min(n, self._pos, cap)
        if n <= 0:
            return np.zeros((4, 0))
        end = self._pos % cap
        if n <= end:
            return self._ring[:, end - n:end].copy()
        return np.concatenate([self._ring[:, cap - (n - end):], self._ring[:, :end]], axis=1)


# ── Stub Neural Pipeline (when neural_pipeline not installed) ────

class _StubEEGNetExtractor:
    """Fallback: returns random features shaped like EEGNet output."""
    def extract(self, window: np.ndarray) -> np.ndarray:
        return np.random.randn(32).astype(np.float32) * 0.1


class _StubBrainStateHMM:
    """Fallback: simple rule-based HMM state from brain metrics."""
    def __init__(self):
        self.state = "browsing"
        self.probs = {s: 0.2 for s in HMM_STATES}
        self.probs["browsing"] = 0.6

    def update(self, brain_state: BrainState, features: np.ndarray):
        eng = brain_state.engagement
        foc = brain_state.focus
        rel = brain_state.relaxation
        err = brain_state.error_response

        # Simple rule-based state assignment
        if err > 0.5:
            self.state = "error"
        elif rel > 0.6 and eng < 0.3:
            self.state = "resting"
        elif eng > 0.7 and foc > 0.5:
            self.state = "intending"
        elif foc > 0.4 and eng > 0.4:
            self.state = "searching"
        else:
            self.state = "browsing"

        # Simulate probability distribution around the active state
        self.probs = {s: 0.05 for s in HMM_STATES}
        self.probs[self.state] = 0.7
        remaining = 0.25
        others = [s for s in HMM_STATES if s != self.state]
        for s in others:
            self.probs[s] = remaining / len(others)

        return self.state, self.probs


class _StubErrPDetector:
    """Fallback: uses raw error_response from BrainState."""
    def detect(self, window: np.ndarray) -> tuple[bool, float]:
        # Check last 200ms of data for ErrP-like waveform
        if window.shape[1] < 50:
            return False, 0.0
        segment = window[:, -int(EEG_SR * 0.3):]
        # Simple peak detection as proxy
        peak = float(np.max(np.abs(segment)))
        prob = min(1.0, peak / 100.0)
        return prob > 0.65, prob


class _StubBayesianIntentAccumulator:
    """Fallback: simple dwell-based probability accumulation."""
    def __init__(self):
        self._elements: dict[str, dict] = {}
        self._decay_rate = 0.95

    def update(self, element_id: str, engagement: float, fixation_duration: float,
               hmm_state: str, features: np.ndarray | None = None):
        if element_id not in self._elements:
            self._elements[element_id] = {"prob": 0.0, "samples": 0}

        entry = self._elements[element_id]

        # Evidence strength from engagement and fixation
        evidence = 0.0
        if fixation_duration > 0.2:
            evidence += 0.05 * engagement
        if hmm_state == "intending":
            evidence += 0.03
        if fixation_duration > 0.5:
            evidence += 0.02

        # Bayesian update: P_new = P_old + evidence * (1 - P_old)
        p = entry["prob"]
        p = p + evidence * (1.0 - p)
        entry["prob"] = min(0.99, p)
        entry["samples"] += 1

        # Decay all other elements
        for eid, e in self._elements.items():
            if eid != element_id:
                e["prob"] *= self._decay_rate

    def get_probability(self, element_id: str) -> float:
        if element_id not in self._elements:
            return 0.0
        return self._elements[element_id]["prob"]

    def get_commitment_level(self, element_id: str) -> str:
        p = self.get_probability(element_id)
        if p >= 0.9:
            return "execute"
        elif p >= 0.7:
            return "committed"
        elif p >= 0.4:
            return "considering"
        elif p >= 0.15:
            return "aware"
        return "none"

    def get_top_element(self) -> tuple[str | None, float]:
        if not self._elements:
            return None, 0.0
        top_id = max(self._elements, key=lambda k: self._elements[k]["prob"])
        return top_id, self._elements[top_id]["prob"]

    def reset(self, element_id: str):
        if element_id in self._elements:
            self._elements[element_id] = {"prob": 0.0, "samples": 0}

    def reset_all(self):
        self._elements.clear()


class _StubNeuralUCBBandit:
    """Fallback: UCB1 bandit for action selection with ErrP-based reward."""
    def __init__(self, actions: list[str]):
        self.actions = actions
        self._counts = {a: 0 for a in actions}
        self._rewards = {a: 0.0 for a in actions}
        self._total = 0
        self.exploration_rate = 1.0

    def select_action(self, context: dict) -> tuple[str, float]:
        element_role = context.get("element_role", "")
        # Role-based priors
        role_priors = {
            "link": "click", "button": "click", "textbox": "type_text",
            "checkbox": "click", "combobox": "click", "searchbox": "type_text",
            "tab": "click", "menuitem": "click",
        }

        # If we have a strong prior, use it most of the time
        prior_action = role_priors.get(element_role)
        if prior_action and prior_action in self.actions:
            if self._total < 10 or np.random.random() < 0.8:
                return prior_action, 0.85

        # UCB1 selection
        if self._total < len(self.actions):
            # Explore each action at least once
            for a in self.actions:
                if self._counts[a] == 0:
                    return a, 0.5

        best_action = self.actions[0]
        best_ucb = -float("inf")
        for a in self.actions:
            if self._counts[a] == 0:
                return a, 0.5
            avg = self._rewards[a] / self._counts[a]
            ucb = avg + self.exploration_rate * np.sqrt(
                2 * np.log(self._total) / self._counts[a]
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        confidence = self._rewards.get(best_action, 0.0)
        if self._counts.get(best_action, 0) > 0:
            confidence = confidence / self._counts[best_action]
        return best_action, float(np.clip(confidence, 0.1, 0.95))

    def update(self, action: str, reward: float):
        if action in self._counts:
            self._counts[action] += 1
            self._rewards[action] += reward
            self._total += 1
            # Decay exploration
            self.exploration_rate = max(0.1, 1.0 / (1 + self._total * 0.01))

    def get_stats(self) -> dict:
        stats = {}
        for a in self.actions:
            c = self._counts[a]
            r = self._rewards[a]
            stats[a] = {
                "count": c,
                "avg_reward": round(r / c, 3) if c > 0 else 0.0,
                "total_reward": round(r, 3),
            }
        stats["_exploration_rate"] = round(self.exploration_rate, 3)
        stats["_total_actions"] = self._total
        return stats


# ── Main BCI Server ──────────────────────────────────────────────

class AxiomV2:
    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                text=True, timeout=5
            )
            for line in out.splitlines():
                if "UI Looks like:" in line:
                    parts = line.split()
                    w, h = int(parts[3]), int(parts[5])
                    print(f"[SCREEN] Detected {w}x{h} (effective)", flush=True)
                    return w, h
                if "Resolution" in line and "x" in line:
                    parts = line.split()
                    idx = parts.index("x")
                    w, h = int(parts[idx - 1]), int(parts[idx + 1])
                    print(f"[SCREEN] Detected {w}x{h}", flush=True)
                    return w, h
        except Exception:
            pass
        print("[SCREEN] Using default 1440x900", flush=True)
        return 1440, 900

    def __init__(self, sim=False):
        self.phase = "startup"
        self.sim = sim

        # Core components
        self.eeg = EEGSource(sim=sim)
        self.brain_engine = BrainStateEngine()
        screen_w, screen_h = self._detect_screen_size()
        self.gaze = GazeTracker(screen_w=screen_w, screen_h=screen_h)
        self.controller = UniversalController()
        self.fixation = FixationDetector()
        self.llm = None

        # Neural pipeline components
        if HAS_NEURAL_PIPELINE:
            self.eegnet = EEGNetExtractor()
            self.hmm = BrainStateHMM()
            self.errp_detector = ErrPDetector()
            self.intent_acc = BayesianIntentAccumulator()
            self.bandit = NeuralUCBBandit()
        else:
            print("[NEURAL] Using stub neural pipeline (install neural_pipeline for full version)", flush=True)
            self.eegnet = _StubEEGNetExtractor()
            self.hmm = _StubBrainStateHMM()
            self.errp_detector = _StubErrPDetector()
            self.intent_acc = _StubBayesianIntentAccumulator()
            self.bandit = _StubNeuralUCBBandit(SUPPORTED_ACTIONS)

        # State
        self.brain_state = BrainState()
        self.gaze_result = GazeResult()
        self.eegnet_features: np.ndarray = np.zeros(32)
        self.hmm_state = "browsing"
        self.hmm_probs: dict[str, float] = {s: 0.2 for s in HMM_STATES}
        self.page_elements: list[WebElement] = []
        self.focused_element: WebElement | None = None
        self.page_url = ""
        self.page_title = ""
        self.page_domain = ""

        self._action_log: list[dict] = []
        self._last_action_time = 0.0
        self._errp_monitor_start = 0.0
        self._errp_monitoring = False
        self._errp_action_entry: dict | None = None
        self._last_bandit_context: np.ndarray | None = None

        self.thresholds = {
            "fixation_dwell_ms": 800,
            "engagement_act": 0.55,
            "intent_execute": 0.90,
            "intent_committed": 0.70,
            "errp_threshold": 0.65,
            "cooldown_ms": 1500,
        }

    async def start(self):
        self.phase = "startup"
        await self._broadcast_system("Starting EEG...")
        self.eeg.start()

        await self._broadcast_system("Starting gaze tracker...")
        try:
            self.gaze.start()
            if self.gaze.load_calibration("data/gaze_calibration.json"):
                await self._broadcast_system("Loaded saved gaze calibration")
        except Exception as e:
            await self._broadcast_system(f"Gaze tracker error: {e} -- continuing without gaze")

        await self._broadcast_system("Launching browser...")
        try:
            await self.controller.start()
            if TEST_MODE:
                await self.controller.navigate_to("http://localhost:8080")
                await self._broadcast_system("Browser ready — loaded test site")
            else:
                await self._broadcast_system("Browser ready")
        except Exception as e:
            await self._broadcast_system(f"Browser error: {e}")

        if USE_LLM:
            try:
                self.llm = AxiomLLM()
                await self._broadcast_system("LLM connected (AWS Bedrock)")
            except Exception as e:
                await self._broadcast_system(f"LLM error: {e}")

        self.phase = "live"
        await self._broadcast_system("System ready -- live mode")

    async def stop(self):
        self.eeg.stop()
        self.gaze.stop()
        await self.controller.stop()

    # ── Main Loops ─────────────────────────────────────────────

    async def eeg_loop(self):
        """50ms: Pull EEG samples into ring buffer."""
        while True:
            self.eeg.pull()
            await asyncio.sleep(0.05)

    async def brain_loop(self):
        """2Hz: Process EEG window -> EEGNet features + BrainState + HMM state."""
        while True:
            window = self.eeg.get_window(1.0)
            if window.shape[1] >= EEG_SR:
                # Basic brain state from band powers
                self.brain_state = self.brain_engine.process(window, timestamp=time.time())

                # EEGNet feature extraction
                if HAS_NEURAL_PIPELINE:
                    self.eegnet_features = self.eegnet.extract_features(window)
                else:
                    self.eegnet_features = self.eegnet.extract(window)

                # HMM state update
                if HAS_NEURAL_PIPELINE:
                    band_ratios = np.array([
                        self.brain_state.engagement, self.brain_state.focus,
                        self.brain_state.relaxation, self.brain_state.cognitive_load,
                        self.brain_state.valence, self.brain_state.error_response,
                    ], dtype=np.float32)
                    hmm_obs = np.concatenate([self.eegnet_features, band_ratios])
                    hmm_result = self.hmm.update(hmm_obs)
                    self.hmm_state = hmm_result["state"]
                    self.hmm_probs = hmm_result["probabilities"]
                else:
                    self.hmm_state, self.hmm_probs = self.hmm.update(
                        self.brain_state, self.eegnet_features
                    )

                await self._broadcast("brain", {
                    "engagement": self.brain_state.engagement,
                    "focus": self.brain_state.focus,
                    "relaxation": self.brain_state.relaxation,
                    "cognitive_load": self.brain_state.cognitive_load,
                    "valence": self.brain_state.valence,
                    "jaw_clench": self.brain_state.jaw_clench,
                    "error_response": self.brain_state.error_response,
                    "hmm_state": self.hmm_state,
                    "hmm_probs": self.hmm_probs,
                    "eegnet_features": self.eegnet_features[:8].tolist(),
                })

                # Jaw clench: immediate action (bypass accumulator)
                if self.brain_state.jaw_clench:
                    await self._handle_clench()

            await asyncio.sleep(1.0 / BRAIN_HZ)

    async def gaze_loop(self):
        """15Hz: Capture gaze -> fixation detection -> element lookup."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                self.gaze_result = await loop.run_in_executor(None, self.gaze.process_frame)
            except Exception:
                await asyncio.sleep(0.1)
                continue

            if self.gaze_result.calibrated:
                sx, sy = self.gaze_result.screen_x, self.gaze_result.screen_y
                self.fixation.update(sx, sy, time.time())

                vx, vy = self.controller.screen_to_viewport(sx, sy)
                self.focused_element = self.controller.element_at_viewport(vx, vy)

                await self._broadcast("gaze", {
                    "screen_x": sx,
                    "screen_y": sy,
                    "viewport_x": vx,
                    "viewport_y": vy,
                    "fixating": self.fixation.fixating,
                    "fixation_duration": self.fixation.fixation_duration,
                    "element": {
                        "id": self.focused_element.id,
                        "role": self.focused_element.role,
                        "name": self.focused_element.name,
                    } if self.focused_element else None,
                    "confidence": self.gaze_result.confidence,
                })

            await asyncio.sleep(1.0 / GAZE_HZ)

    async def page_scan_loop(self):
        """0.5Hz: Scan page accessibility tree for elements."""
        while True:
            try:
                ctx = await self.controller.get_context()
                self.page_elements = ctx.elements
                self.page_url = ctx.url
                self.page_title = ctx.title
                self.page_domain = ctx.domain

                await self._broadcast("page", {
                    "url": ctx.url,
                    "title": ctx.title,
                    "domain": ctx.domain,
                    "element_count": len(ctx.elements),
                    "elements": [
                        {
                            "id": e.id,
                            "role": e.role,
                            "name": e.name[:80],
                            "bbox": e.bbox,
                        }
                        for e in ctx.elements[:40]
                    ],
                })
            except Exception as e:
                await self._broadcast_system(f"Page scan error: {e}")
            await asyncio.sleep(1.0 / SCAN_HZ)

    async def intent_loop(self):
        """5Hz: Update Bayesian intent accumulator -> check commitment -> maybe execute."""
        while True:
            if self.phase != "live":
                await asyncio.sleep(0.5)
                continue

            now = time.time()
            if now - self._last_action_time < self.thresholds["cooldown_ms"] / 1000:
                await asyncio.sleep(1.0 / INTENT_HZ)
                continue

            # 1. Get fixated element from gaze
            element = self.focused_element
            if not self.fixation.fixating or not element:
                # Broadcast empty intent to clear UI ring
                await self._broadcast("intent", {
                    "element_id": None,
                    "element_name": None,
                    "probability": 0.0,
                    "commitment_level": "none",
                    "action_prediction": None,
                    "action_confidence": 0.0,
                })
                await asyncio.sleep(1.0 / INTENT_HZ)
                continue

            # 2. Get brain state
            engagement = self.brain_state.engagement
            faa = 0.0
            if hasattr(self.brain_state, 'valence'):
                faa = (self.brain_state.valence - 0.5) * 2.0
            dwell = self.fixation.fixation_duration

            # 3. Update Bayesian intent accumulator with evidence
            if HAS_NEURAL_PIPELINE:
                self.intent_acc.update(
                    fixated_element_id=element.id,
                    fixation_duration=dwell,
                    engagement=engagement,
                    faa=faa,
                    hmm_state=self.hmm_state,
                    hmm_confidence=self.hmm_probs.get(self.hmm_state, 0.5),
                )
                _, prob = self.intent_acc.get_top_intent()
                prob = prob if _ == element.id else self.intent_acc._element_beliefs.get(element.id, 0.0)
            else:
                self.intent_acc.update(
                    element_id=element.id,
                    engagement=engagement,
                    fixation_duration=dwell,
                    hmm_state=self.hmm_state,
                    features=self.eegnet_features,
                )
                prob = self.intent_acc.get_probability(element.id)

            # 4. Get commitment level for top element
            commitment = self.intent_acc.get_commitment_level(element.id)

            # 5. Get action prediction from bandit
            if HAS_NEURAL_PIPELINE:
                ctx = self.bandit.build_context(
                    self.eegnet_features, self.hmm_state, element.role,
                    engagement, faa, dwell)
                self._last_bandit_context = ctx
                action_pred, action_conf = self.bandit.select_action(ctx)
            else:
                action_pred, action_conf = self.bandit.select_action({
                    "element_role": element.role,
                    "hmm_state": self.hmm_state,
                    "engagement": engagement,
                })

            # 6. Broadcast intent state for UI commitment ring
            await self._broadcast("intent", {
                "element_id": element.id,
                "element_name": element.name[:80],
                "probability": round(prob, 3),
                "commitment_level": commitment,
                "action_prediction": action_pred,
                "action_confidence": round(action_conf, 3),
            })

            # 7. If commitment >= 'execute' (P > 0.9): act
            if commitment == "execute" and engagement >= self.thresholds["engagement_act"]:
                await self._execute_action(
                    action_pred, element, action_conf,
                    f"Bayesian P={prob:.2f}, commitment={commitment}, "
                    f"HMM={self.hmm_state}, engagement={engagement:.2f}",
                )
                # Reset intent for this element after execution
                if HAS_NEURAL_PIPELINE:
                    self.intent_acc.reset_element(element.id)
                else:
                    self.intent_acc.reset(element.id)

            await asyncio.sleep(1.0 / INTENT_HZ)

    async def errp_loop(self):
        """Monitors for ErrP in 200-500ms window after each action."""
        while True:
            if not self._errp_monitoring:
                await asyncio.sleep(0.05)
                continue

            now = time.time()
            elapsed = now - self._errp_monitor_start

            # Wait for 200ms before checking
            if elapsed < 0.2:
                await asyncio.sleep(0.02)
                continue

            # Check in the 200-500ms window
            if elapsed <= 0.5:
                window = self.eeg.get_window(0.3)
                if HAS_NEURAL_PIPELINE:
                    prob = self.errp_detector.detect(window, self._errp_monitor_start)
                    detected = prob > 0.65
                else:
                    detected, prob = self.errp_detector.detect(window)

                if detected:
                    self._errp_monitoring = False
                    # Negative reward to bandit
                    if self._errp_action_entry:
                        action = self._errp_action_entry.get("action", "none")
                        if HAS_NEURAL_PIPELINE and self._last_bandit_context is not None:
                            self.bandit.update(self._last_bandit_context, action, -1.0)
                        else:
                            self.bandit.update(action, -1.0)
                        await self._broadcast("errp", {
                            "detected": True,
                            "probability": round(prob, 3),
                            "undoing": action,
                            "target": self._errp_action_entry.get("target", ""),
                        })
                        await self._broadcast_system(
                            f"ErrP detected (P={prob:.2f}) -- undoing {action}"
                        )
                        # Attempt undo
                        try:
                            await self.controller.press_key("Meta+z")
                        except Exception:
                            pass
                    self._errp_action_entry = None
                    continue

                await asyncio.sleep(0.02)
                continue

            # Window expired without ErrP -> positive reward
            if elapsed > 0.5:
                self._errp_monitoring = False
                if self._errp_action_entry:
                    action = self._errp_action_entry.get("action", "none")
                    if HAS_NEURAL_PIPELINE and self._last_bandit_context is not None:
                        self.bandit.update(self._last_bandit_context, action, 1.0)
                    else:
                        self.bandit.update(action, 1.0)
                    await self._broadcast("errp", {
                        "detected": False,
                        "probability": 0.0,
                        "undoing": None,
                        "target": None,
                    })
                self._errp_action_entry = None

            await asyncio.sleep(0.05)

    # ── Action Execution ─────────────────────────────────────────

    async def _execute_action(self, action: str, element: WebElement,
                               confidence: float, reason: str):
        self._last_action_time = time.time()
        entry = {
            "action": action,
            "target": element.name[:80],
            "element_role": element.role,
            "confidence": round(confidence, 3),
            "reason": reason,
            "timestamp": time.time(),
            "brain": {
                "engagement": self.brain_state.engagement,
                "focus": self.brain_state.focus,
                "valence": self.brain_state.valence,
                "hmm_state": self.hmm_state,
            },
        }

        try:
            if action == "click":
                await self.controller.click_element(element)
                entry["result"] = "clicked"
            elif action == "scroll_down":
                await self.controller.scroll("down")
                entry["result"] = "scrolled down"
            elif action == "scroll_up":
                await self.controller.scroll("up")
                entry["result"] = "scrolled up"
            elif action == "back":
                await self.controller.navigate_back()
                entry["result"] = "navigated back"
            elif action == "forward":
                await self.controller.navigate_forward()
                entry["result"] = "navigated forward"
            elif action == "open_tab":
                await self.controller.open_in_new_tab(element)
                entry["result"] = "opened in new tab"
            elif action == "close_tab":
                await self.controller.close_tab()
                entry["result"] = "tab closed"
            elif action == "type_text":
                if self.llm:
                    # Let LLM decide what to type based on context
                    page_text = await self.controller.get_page_text()
                    # For now, just click to focus
                    await self.controller.click_element(element)
                    entry["result"] = "focused for typing"
                else:
                    await self.controller.click_element(element)
                    entry["result"] = "focused for typing"
            else:
                entry["result"] = "no_op"
                return
        except Exception as e:
            entry["result"] = f"error: {e}"

        self._action_log.append(entry)
        await self._broadcast("action", entry)

        # Start ErrP monitoring window
        self._errp_monitoring = True
        self._errp_monitor_start = time.time()
        self._errp_action_entry = entry

    async def _handle_clench(self):
        """Jaw clench = immediate action, bypasses accumulator."""
        if not self.focused_element:
            return
        now = time.time()
        if now - self._last_action_time < self.thresholds["cooldown_ms"] / 1000:
            return

        element = self.focused_element
        # Default clench action based on role
        role = element.role
        if role in ("link", "button", "checkbox", "tab", "menuitem",
                     "radio", "switch", "option", "treeitem"):
            action = "click"
        elif role in ("textbox", "searchbox", "combobox"):
            action = "click"  # Focus the field
        else:
            action = "click"

        await self._execute_action(
            action, element, 0.95, "Jaw clench confirmation"
        )
        if HAS_NEURAL_PIPELINE:
            self.intent_acc.reset_element(element.id)
        else:
            self.intent_acc.reset(element.id)

    # ── Calibration Handlers ─────────────────────────────────────

    async def start_gaze_calibration(self):
        self.phase = "gaze_cal"
        targets = self.gaze.get_calibration_targets(9)
        await self._broadcast("calibration", {
            "type": "gaze_start",
            "targets": [{"x": t[0], "y": t[1]} for t in targets],
            "total": len(targets),
        })

    async def add_gaze_point(self, screen_x: float, screen_y: float):
        ready = self.gaze.add_calibration_point(screen_x, screen_y)
        if ready:
            self.gaze.calibrate()
            self.gaze.save_calibration("data/gaze_calibration.json")
            self.phase = "live"
            await self._broadcast("calibration", {"type": "gaze_done", "success": True})
        else:
            await self._broadcast("calibration", {"type": "gaze_point_added"})

    async def start_eeg_calibration(self):
        self.phase = "calibrate"
        await self._broadcast("calibration", {"type": "eeg_start"})
        await self._broadcast_system("EEG calibration started -- perform actions while we record")

    async def end_eeg_calibration(self):
        self.phase = "live"
        await self._broadcast("calibration", {"type": "eeg_done", "success": True})
        await self._broadcast_system("EEG calibration complete")

    # ── WebSocket ────────────────────────────────────────────────

    async def _broadcast(self, msg_type: str, data: dict):
        msg = json.dumps({"type": msg_type, "timestamp": time.time(), **data})
        dead = set()
        for ws in list(clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        clients.difference_update(dead)

    async def _broadcast_system(self, text: str):
        print(f"  [{self.phase.upper()}] {text}", flush=True)
        await self._broadcast("system", {"status": text, "phase": self.phase})

    async def handle_ws(self, ws):
        clients.add(ws)
        print(f"Dashboard connected ({len(clients)})", flush=True)
        await ws.send(json.dumps({
            "type": "init",
            "phase": self.phase,
            "sim": self.sim,
            "thresholds": self.thresholds,
            "action_log": self._action_log[-20:],
            "supported_actions": SUPPORTED_ACTIONS,
        }))
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    await self._handle_command(data)
                except json.JSONDecodeError:
                    pass
        finally:
            clients.discard(ws)

    async def _handle_command(self, data: dict):
        cmd = data.get("command")
        if cmd == "start_gaze_cal":
            await self.start_gaze_calibration()
        elif cmd == "gaze_point":
            await self.add_gaze_point(data["x"], data["y"])
        elif cmd == "start_eeg_cal":
            await self.start_eeg_calibration()
        elif cmd == "end_eeg_cal":
            await self.end_eeg_calibration()
        elif cmd == "set_phase":
            self.phase = data["phase"]
            await self._broadcast_system(f"Phase -> {self.phase}")
        elif cmd == "set_threshold":
            key, val = data["key"], float(data["value"])
            if key in self.thresholds:
                self.thresholds[key] = val
        elif cmd == "reset_bandit":
            self.bandit = (
                NeuralUCBBandit(SUPPORTED_ACTIONS) if HAS_NEURAL_PIPELINE
                else _StubNeuralUCBBandit(SUPPORTED_ACTIONS)
            )
            await self._broadcast_system("Bandit reset")
        elif cmd == "reset_intent":
            self.intent_acc.reset_all()
            await self._broadcast_system("Intent accumulator reset")


# ── Entry Point ──────────────────────────────────────────────────

async def main():
    bci = AxiomV2(sim=SIM_MODE)

    os.makedirs("data", exist_ok=True)

    server = await websockets.serve(bci.handle_ws, "127.0.0.1", 8765)
    print(f"Axiom v2 server on ws://127.0.0.1:8765", flush=True)
    print(f"  Sim mode: {SIM_MODE}", flush=True)
    print(f"  LLM: {USE_LLM}", flush=True)
    print(f"  Neural pipeline: {HAS_NEURAL_PIPELINE}", flush=True)

    await bci.start()

    await asyncio.gather(
        bci.eeg_loop(),
        bci.brain_loop(),
        bci.gaze_loop(),
        bci.page_scan_loop(),
        bci.intent_loop(),
        bci.errp_loop(),
        server.wait_closed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
