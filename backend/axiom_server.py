#!/usr/bin/env python3
"""Axiom Server — full brain-computer interface pipeline.

Layers:
  1. BrainFlow → Muse S EEG at 256Hz
  2. BrainStateEngine → cognitive states at 10Hz
  3. GazeTracker → screen position at 30Hz (when calibrated)
  4. OSControl → screen context
  5. AxiomAgent → decisions (threshold-based + LLM fallback)
  6. LLM Engine → self-improvement every 20 actions / 5 min

WebSocket messages:
  - "eeg"     (20Hz): filtered waveforms + band powers
  - "state"   (1Hz):  29-dim RL state vector
  - "brain"   (2Hz):  classified brain states
  - "action"  (event): agent decisions
  - "learning" (5s):  accuracy curve + decision log
"""

import asyncio
import json
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

import websockets
from features import extract_state, BANDS
from brain_state import BrainStateEngine
from axiom_agent import AxiomAgent, ScreenContext
from os_control import OSControl
from overlay_ctrl import OverlayController

BOARD_ID = BoardIds.MUSE_S_BOARD.value
EEG_SR = 256
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Globals
board = None
clients: set = set()
eeg_ring = np.zeros((4, EEG_SR * 10))
ring_pos = 0
total_samples = 0
prev_bands = None

# Axiom modules
brain_engine = BrainStateEngine()
agent = AxiomAgent()
os_ctrl = OSControl()
overlay = OverlayController()
llm = None  # initialized lazily

# Current state
current_brain = None
current_gaze = None
current_screen = None

USE_GAZE = False
USE_LLM = False
USE_OS_ACTIONS = False
USE_OVERLAY = True


def connect_muse():
    global board
    params = BrainFlowInputParams()
    params.timeout = 5
    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)
    print("Scanning for Muse S...", flush=True)
    board.prepare_session()
    print("Connected!", flush=True)
    board.start_stream(num_samples=450000)
    print("Streaming EEG at 256 Hz", flush=True)


def filter_channel(data, sr=EEG_SR):
    out = data.copy()
    if len(out) < 12:
        return out
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, sr, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, sr, NoiseTypes.SIXTY.value)
    return out


def get_ring_slice(n_samples):
    cap = eeg_ring.shape[1]
    n = min(n_samples, total_samples, cap)
    if n <= 0:
        return np.zeros((4, 0))
    end = ring_pos % cap
    if n <= end:
        return eeg_ring[:, end - n:end].copy()
    return np.concatenate([eeg_ring[:, cap - (n - end):], eeg_ring[:, :end]], axis=1)


def compute_bands_display(filtered_4ch):
    band_list = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                 ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]
    result = {name: 0.0 for name, _, _ in band_list}
    count = 0
    for ch in range(4):
        d = filtered_4ch[ch]
        if len(d) < EEG_SR:
            continue
        nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
        psd = DataFilter.get_psd_welch(d, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
        for name, lo, hi in band_list:
            result[name] += float(DataFilter.get_band_power(psd, lo, hi))
        count += 1
    if count > 0:
        for name in result:
            result[name] /= count
    return result


def build_screen_context() -> ScreenContext:
    """Build ScreenContext from OS state + cursor position (gaze proxy)."""
    ctx = ScreenContext()
    try:
        state = os_ctrl.get_screen_state()
        ctx.active_app = state.active_app
        if state.focused_window:
            ctx.gaze_target = state.focused_window
        ctx.available_actions = state.dock_apps[:6]
        cx, cy = os_ctrl.get_cursor_position()
        ctx.gaze_region = os_ctrl._classify_gaze_region(float(cx), float(cy))
    except Exception:
        pass
    return ctx


async def broadcast(msg: str):
    dead = set()
    for ws in list(clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    for d in dead:
        clients.discard(d)


async def stream_loop():
    global ring_pos, total_samples, prev_bands, current_brain

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    state_tick = 0
    brain_tick = 0
    learning_tick = 0
    improve_check = 0

    while True:
        raw = board.get_board_data(num_samples=128)
        if raw.shape[1] == 0:
            await asyncio.sleep(0.02)
            continue

        n = raw.shape[1]
        eeg = raw[eeg_channels[:4], :]
        now = time.time()

        # Ring buffer
        cap = eeg_ring.shape[1]
        for i in range(n):
            eeg_ring[:, ring_pos % cap] = eeg[:, i]
            ring_pos += 1
        total_samples += n

        # --- EEG display (20Hz) ---
        segment = get_ring_slice(EEG_SR * 2)
        filtered = np.zeros_like(segment)
        for ch in range(4):
            filtered[ch] = filter_channel(segment[ch])

        bands = compute_bands_display(filtered)
        step = 4
        eeg_msg = json.dumps({
            "type": "eeg",
            "channels": CHANNELS,
            "data": filtered[:, ::step].tolist(),
            "bands": bands,
            "sampleRate": EEG_SR // step,
            "totalSamples": total_samples,
            "timestamp": now,
        })
        await broadcast(eeg_msg)

        # --- RL state (1Hz) ---
        state_tick += n
        if state_tick >= EEG_SR:
            state_tick = 0
            state_seg = get_ring_slice(EEG_SR * 2)
            result = extract_state(state_seg, prev_bands)
            prev_bands = result["bands"]
            await broadcast(json.dumps({
                "type": "state",
                **result,
                "timestamp": now,
                "totalSamples": total_samples,
            }))

        # --- Brain state (2Hz) ---
        brain_tick += n
        if brain_tick >= EEG_SR // 2:
            brain_tick = 0
            brain_seg = get_ring_slice(EEG_SR)
            if brain_seg.shape[1] >= EEG_SR:
                current_brain = brain_engine.process(brain_seg, timestamp=now)

                await broadcast(json.dumps({
                    "type": "brain",
                    "engagement": current_brain.engagement,
                    "focus": current_brain.focus,
                    "relaxation": current_brain.relaxation,
                    "cognitive_load": current_brain.cognitive_load,
                    "valence": current_brain.valence,
                    "jaw_clench": current_brain.jaw_clench,
                    "double_clench": current_brain.double_clench,
                    "context_switch": current_brain.context_switch,
                    "error_response": current_brain.error_response,
                    "timestamp": now,
                }))

                # --- Update overlay ---
                ctx = build_screen_context()
                if USE_OVERLAY and overlay.running:
                    overlay.presence(True, current_brain.engagement)
                    overlay.mode(agent._current_mode)
                    try:
                        cx, cy = os_ctrl.get_cursor_position()
                        overlay.gaze(cx, cy)
                    except Exception:
                        pass
                action = agent.decide(current_brain, ctx, timestamp=now)

                if action.action_type != "none":
                    await broadcast(json.dumps({
                        "type": "action",
                        "action_type": action.action_type,
                        "target": action.target,
                        "confidence": action.confidence,
                        "reason": action.reason,
                        "timestamp": now,
                    }))

                    # Update overlay for actions
                    if USE_OVERLAY and overlay.running:
                        if "intent_ring" in action.action_type:
                            try:
                                ix, iy = os_ctrl.get_cursor_position()
                            except Exception:
                                ix, iy = 0, 0
                            overlay.intent_ring(True, action.confidence, ix, iy)
                        elif action.action_type in ("click", "open"):
                            overlay.flash("green")
                            overlay.intent_ring(False)
                        elif action.action_type == "undo":
                            overlay.flash("red")

                    # Execute OS action if enabled
                    if USE_OS_ACTIONS:
                        execute_action(action)

        # --- Learning metrics (every 5s) ---
        learning_tick += n
        if learning_tick >= EEG_SR * 5:
            learning_tick = 0
            await broadcast(json.dumps({
                "type": "learning",
                "accuracy": agent.accuracy,
                "accuracy_history": agent.accuracy_history[-100:],
                "total_actions": agent.total_actions,
                "undone_actions": agent.undone_actions,
                "thresholds": agent.thresholds,
                "decision_log": agent.get_decision_log(10),
                "mode": agent._current_mode,
                "llm_stats": llm.stats if llm else None,
                "timestamp": now,
            }))

            # --- Self-improvement check ---
            improve_check += 1
            if agent.should_improve() and USE_LLM and llm:
                asyncio.create_task(run_self_improvement())

        await asyncio.sleep(0.05)


def execute_action(action):
    """Actually perform an OS action. Only when USE_OS_ACTIONS is True."""
    if action.action_type == "click" and action.target:
        print(f"  [EXEC] click → {action.target}")
    elif action.action_type == "undo":
        os_ctrl.undo()
        print("  [EXEC] undo (Cmd+Z)")
    elif action.action_type == "scroll_down":
        os_ctrl.scroll(-3)
    elif action.action_type in ("switch_app", "open") and action.target:
        os_ctrl.switch_app(action.target)
        print(f"  [EXEC] switch_app → {action.target}")


async def run_self_improvement():
    """Run the LLM self-improvement step in the background."""
    if not llm:
        return
    try:
        context = agent.get_improvement_context()
        result = llm.review_strategy(context, agent.thresholds)
        if "thresholds" in result:
            agent.apply_improvements(result["thresholds"])
            print(f"  [LLM] Strategy review complete: {result.get('observations', '')[:100]}")
    except Exception as e:
        print(f"  [LLM] Error: {e}")


async def handler(ws):
    clients.add(ws)
    print(f"Client connected ({len(clients)} total)", flush=True)
    try:
        await ws.send(json.dumps({
            "type": "info",
            "channels": CHANNELS,
            "sampleRate": EEG_SR,
            "bands": [name for name, _, _ in BANDS],
            "stateSize": 29,
            "features": ["brain_state", "agent", "learning"],
            "gaze_enabled": USE_GAZE,
            "llm_enabled": USE_LLM,
            "os_actions_enabled": USE_OS_ACTIONS,
        }))
        async for msg in ws:
            try:
                data = json.loads(msg)
                await handle_client_message(data)
            except json.JSONDecodeError:
                pass
    finally:
        clients.discard(ws)
        print(f"Client disconnected ({len(clients)} total)", flush=True)


async def handle_client_message(data: dict):
    """Handle commands from the frontend."""
    cmd = data.get("command")
    if cmd == "enable_os_actions":
        global USE_OS_ACTIONS
        USE_OS_ACTIONS = data.get("enabled", False)
        print(f"  OS actions: {'ON' if USE_OS_ACTIONS else 'OFF'}")
    elif cmd == "enable_llm":
        global USE_LLM, llm
        USE_LLM = data.get("enabled", False)
        if USE_LLM and llm is None:
            from llm_engine import AxiomLLM
            llm = AxiomLLM()
        print(f"  LLM: {'ON' if USE_LLM else 'OFF'}")
    elif cmd == "record_outcome":
        outcome = data.get("outcome", "confirmed")
        agent.record_outcome(outcome)
    elif cmd == "adjust_threshold":
        key = data.get("key")
        value = data.get("value")
        if key in agent.thresholds:
            agent.thresholds[key] = float(value)


async def main():
    import sys
    global USE_GAZE, USE_LLM, USE_OS_ACTIONS, USE_OVERLAY

    if "--gaze" in sys.argv:
        USE_GAZE = True
    if "--llm" in sys.argv:
        USE_LLM = True
        from llm_engine import AxiomLLM
        global llm
        llm = AxiomLLM()
    if "--no-actions" in sys.argv:
        USE_OS_ACTIONS = False
    if "--no-overlay" in sys.argv:
        USE_OVERLAY = False

    # Start overlay
    if USE_OVERLAY:
        try:
            overlay.start()
            overlay.presence(True, 0.3)
            overlay.mode("passive")
            overlay.status("AXIOM · CONNECTING...")
        except Exception as e:
            print(f"  [OVERLAY] Failed to start: {e}")
            USE_OVERLAY = False

    connect_muse()

    if USE_OVERLAY and overlay.running:
        overlay.status("AXIOM · STREAMING")
        overlay.mode("tracking")

    port = 8080
    server = await websockets.serve(handler, "127.0.0.1", port)
    print(f"Axiom server on ws://127.0.0.1:{port}", flush=True)
    print(f"  Overlay: {'ON' if USE_OVERLAY else 'OFF'}", flush=True)
    print(f"  Gaze:    {'ON' if USE_GAZE else 'OFF'}", flush=True)
    print(f"  LLM:     {'ON' if USE_LLM else 'OFF'}", flush=True)
    print(f"  Actions: {'ON' if USE_OS_ACTIONS else 'OFF'}", flush=True)

    asyncio.create_task(stream_loop())
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
