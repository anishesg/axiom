#!/usr/bin/env python3
"""Gmail BCI — Brain-Computer Interface for Gmail.

Orchestrates:
  1. Muse S EEG (BrainFlow) → brain state at 2Hz
  2. Webcam gaze tracker (MediaPipe) → screen position at 15Hz
  3. Playwright browser → Gmail control + element positions
  4. Calibration engine → learns EEG→intent mappings
  5. LLM (AWS Bedrock Claude) → smart action decisions + reply drafting

Phases:
  - startup: connecting hardware + launching browser
  - gaze_cal: 9-point gaze calibration
  - baseline: 2-min EEG baseline recording
  - calibrate: user performs actions, system records EEG patterns
  - live: system predicts and acts from brain + gaze
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
from gmail_controller import GmailController, GmailElement
from calibration import CalibrationEngine
from llm_engine import AxiomLLM

SIM_MODE = "--sim" in sys.argv
USE_LLM = "--llm" in sys.argv or os.environ.get("USE_LLM")
EEG_SR = 256
GAZE_HZ = 15
BRAIN_HZ = 2
SCAN_HZ = 0.5

clients: set = set()

# ── Fixation Detector ──────────────────────────────────────────────

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


# ── EEG Source (real or simulated) ─────────────────────────────────

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


# ── Main BCI Server ───────────────────────────────────────────────

class GmailBCI:
    def __init__(self, sim=False):
        self.phase = "startup"
        self.sim = sim

        self.eeg = EEGSource(sim=sim)
        self.brain_engine = BrainStateEngine()
        self.gaze = GazeTracker(screen_w=2560, screen_h=1440)
        self.gmail = GmailController()
        self.calibration = CalibrationEngine()
        self.fixation = FixationDetector()
        self.llm = None

        self.brain_state = BrainState()
        self.gaze_result = GazeResult()
        self.gmail_elements: list[GmailElement] = []
        self.focused_element: GmailElement | None = None

        self._action_log: list[dict] = []
        self._last_action_time = 0.0
        self._intent_ring_start = 0.0
        self._intent_ring_target: str | None = None
        self._errp_window: list[float] = []

        self.thresholds = {
            "fixation_dwell_ms": 800,
            "engagement_act": 0.60,
            "engagement_scroll": 0.30,
            "confidence_auto": 0.75,
            "confidence_clench": 0.50,
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
            await self._broadcast_system(f"Gaze tracker error: {e} — continuing without gaze")

        await self._broadcast_system("Launching Gmail browser...")
        try:
            await self.gmail.start()
            await self._broadcast_system("Gmail ready")
        except Exception as e:
            await self._broadcast_system(f"Gmail error: {e}")

        if USE_LLM:
            try:
                self.llm = AxiomLLM()
                await self._broadcast_system("LLM connected (AWS Bedrock)")
            except Exception as e:
                await self._broadcast_system(f"LLM error: {e}")

        if self.calibration.load():
            await self._broadcast_system("Loaded saved calibration model")

        self.phase = "live"
        await self._broadcast_system("System ready — live mode")

    async def stop(self):
        self.eeg.stop()
        self.gaze.stop()
        await self.gmail.stop()

    # ── Main Loops ─────────────────────────────────────────────

    async def eeg_loop(self):
        while True:
            self.eeg.pull()
            await asyncio.sleep(0.05)

    async def brain_loop(self):
        while True:
            window = self.eeg.get_window(1.0)
            if window.shape[1] >= EEG_SR:
                self.brain_state = self.brain_engine.process(window, timestamp=time.time())
                await self._broadcast("brain", {
                    "engagement": self.brain_state.engagement,
                    "focus": self.brain_state.focus,
                    "relaxation": self.brain_state.relaxation,
                    "cognitive_load": self.brain_state.cognitive_load,
                    "valence": self.brain_state.valence,
                    "jaw_clench": self.brain_state.jaw_clench,
                    "error_response": self.brain_state.error_response,
                })

                if self.brain_state.error_response > self.thresholds["errp_threshold"]:
                    await self._handle_errp()

                if self.brain_state.jaw_clench:
                    await self._handle_clench()

            await asyncio.sleep(1.0 / BRAIN_HZ)

    async def gaze_loop(self):
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

                vx, vy = self.gmail.screen_to_viewport(sx, sy)
                self.focused_element = self.gmail.element_at_viewport(vx, vy)

                await self._broadcast("gaze", {
                    "screen_x": sx,
                    "screen_y": sy,
                    "viewport_x": vx,
                    "viewport_y": vy,
                    "fixating": self.fixation.fixating,
                    "fixation_duration": self.fixation.fixation_duration,
                    "element": {
                        "id": self.focused_element.id,
                        "type": self.focused_element.type,
                        "label": self.focused_element.label,
                    } if self.focused_element else None,
                    "confidence": self.gaze_result.confidence,
                })

            await asyncio.sleep(1.0 / GAZE_HZ)

    async def gmail_scan_loop(self):
        while True:
            try:
                self.gmail_elements = await self.gmail.scan_elements()
                ctx = await self.gmail.get_context()
                await self._broadcast("gmail", {
                    "view": ctx.view,
                    "element_count": len(ctx.elements),
                    "unread_count": ctx.unread_count,
                    "selected_email": ctx.selected_email,
                    "elements": [
                        {"id": e.id, "type": e.type, "label": e.label[:80],
                         "bbox": e.bbox}
                        for e in ctx.elements[:30]
                    ],
                })
            except Exception as e:
                await self._broadcast_system(f"Gmail scan error: {e}")
            await asyncio.sleep(1.0 / SCAN_HZ)

    async def intent_loop(self):
        while True:
            if self.phase != "live":
                await asyncio.sleep(0.5)
                continue

            now = time.time()
            if now - self._last_action_time < self.thresholds["cooldown_ms"] / 1000:
                await asyncio.sleep(0.2)
                continue

            if not self.fixation.fixating or not self.focused_element:
                await asyncio.sleep(0.2)
                continue

            element = self.focused_element
            engagement = self.brain_state.engagement
            dwell = self.fixation.fixation_duration

            intent, confidence = self.calibration.predict_intent(self.brain_state)
            action, action_conf = self.calibration.predict_action(
                self.brain_state,
                {"element_type": element.type, "element_id": element.id,
                 "view": "unknown",
                 **element.metadata}
            )

            dwell_met = dwell * 1000 >= self.thresholds["fixation_dwell_ms"]
            engaged = engagement >= self.thresholds["engagement_act"]

            if dwell_met and engaged and action != "none" and action_conf >= self.thresholds["confidence_auto"]:
                await self._execute_action(action, element, action_conf,
                    f"Dwell {dwell:.1f}s + engagement {engagement:.2f} + intent={intent}")

            elif dwell_met and engaged:
                progress = min(1.0, dwell / (self.thresholds["fixation_dwell_ms"] / 1000 * 2))
                await self._broadcast("intent_ring", {
                    "element_id": element.id,
                    "element_label": element.label,
                    "progress": progress,
                    "intent": intent,
                    "confidence": confidence,
                    "engagement": engagement,
                })

            await asyncio.sleep(0.2)

    # ── Action Execution ──────────────────────────────────────

    async def _execute_action(self, action: str, element: GmailElement,
                               confidence: float, reason: str):
        self._last_action_time = time.time()
        entry = {
            "action": action, "target": element.label[:80],
            "element_type": element.type, "confidence": confidence,
            "reason": reason, "timestamp": time.time(),
            "brain": {"engagement": self.brain_state.engagement,
                      "focus": self.brain_state.focus,
                      "valence": self.brain_state.valence},
        }

        try:
            if action == "open_email" and element.type == "email_row":
                await self.gmail.click_element(element)
                entry["result"] = "opened"
            elif action == "archive":
                await self.gmail.archive_email()
                entry["result"] = "archived"
            elif action == "star" and element.type == "star_button":
                await self.gmail.click_element(element)
                entry["result"] = "starred"
            elif action == "reply":
                if self.llm:
                    body = await self.gmail.get_email_body()
                    draft = self.llm.draft_reply(body[:500])
                    await self.gmail.reply_to_email(draft)
                    entry["result"] = f"replied: {draft[:50]}..."
                else:
                    await self.gmail.reply_to_email("")
                    entry["result"] = "reply opened"
            elif action == "scroll":
                await self.gmail.scroll_inbox("down")
                entry["result"] = "scrolled"
            elif action == "back":
                await self.gmail.back_to_inbox()
                entry["result"] = "back to inbox"
            else:
                entry["result"] = "no_op"
                return
        except Exception as e:
            entry["result"] = f"error: {e}"

        self._action_log.append(entry)
        self.calibration.add_sample(
            self.brain_state, action,
            {"element_type": element.type, **element.metadata},
            {"x": self.gaze_result.screen_x, "y": self.gaze_result.screen_y}
        )

        await self._broadcast("action", entry)

    async def _handle_clench(self):
        if not self.focused_element:
            return
        now = time.time()
        if now - self._last_action_time < self.thresholds["cooldown_ms"] / 1000:
            return

        element = self.focused_element
        action = "none"
        if element.type == "email_row":
            action = "open_email"
        elif element.type == "star_button":
            action = "star"
        elif element.type in ("archive_button", "delete_button", "reply_button",
                                "compose_button", "back_button"):
            action = element.type.replace("_button", "")
        elif element.type == "nav_item":
            action = "open_email"

        if action != "none":
            await self._execute_action(action, element, 0.95, "Jaw clench confirmation")

    async def _handle_errp(self):
        if not self._action_log:
            return
        last = self._action_log[-1]
        if time.time() - last["timestamp"] > 3.0:
            return

        self.calibration.record_feedback(correct=False)
        await self._broadcast("errp", {
            "undoing": last["action"],
            "target": last["target"],
            "error_strength": self.brain_state.error_response,
        })
        await self._broadcast_system(f"ErrP detected — undoing {last['action']}")

        try:
            if last["action"] in ("open_email", "star", "archive"):
                from os_control import OSControl
                OSControl().undo()
        except Exception:
            pass

    # ── Calibration Handlers ──────────────────────────────────

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
        self.calibration.start_calibration_session()
        await self._broadcast("calibration", {"type": "eeg_start"})

    async def end_eeg_calibration(self):
        results = self.calibration.end_calibration_session()
        self.calibration.save()
        self.phase = "live"
        await self._broadcast("calibration", {"type": "eeg_done", "results": results})

    async def record_calibration_action(self, action: str):
        self.calibration.add_sample(
            self.brain_state, action,
            {"element_type": self.focused_element.type if self.focused_element else "none"},
            {"x": self.gaze_result.screen_x, "y": self.gaze_result.screen_y}
        )
        stats = self.calibration.get_calibration_stats()
        await self._broadcast("calibration", {"type": "sample_added", "action": action, "stats": stats})

    # ── WebSocket ─────────────────────────────────────────────

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
            "calibration_stats": self.calibration.get_calibration_stats(),
            "action_log": self._action_log[-20:],
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
        elif cmd == "cal_action":
            await self.record_calibration_action(data["action"])
        elif cmd == "set_phase":
            self.phase = data["phase"]
            await self._broadcast_system(f"Phase → {self.phase}")
        elif cmd == "set_threshold":
            key, val = data["key"], float(data["value"])
            if key in self.thresholds:
                self.thresholds[key] = val
        elif cmd == "feedback":
            self.calibration.record_feedback(data.get("correct", True))
        elif cmd == "retrain":
            results = self.calibration.end_calibration_session()
            self.calibration.save()
            await self._broadcast("calibration", {"type": "retrained", "results": results})


# ── Entry Point ───────────────────────────────────────────────────

async def main():
    bci = GmailBCI(sim=SIM_MODE)

    os.makedirs("data", exist_ok=True)

    server = await websockets.serve(bci.handle_ws, "127.0.0.1", 8765)
    print(f"Gmail BCI server on ws://127.0.0.1:8765", flush=True)
    print(f"  Sim mode: {SIM_MODE}", flush=True)
    print(f"  LLM: {USE_LLM}", flush=True)

    await bci.start()

    await asyncio.gather(
        bci.eeg_loop(),
        bci.brain_loop(),
        bci.gaze_loop(),
        bci.gmail_scan_loop(),
        bci.intent_loop(),
        server.wait_closed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
