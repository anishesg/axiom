#!/usr/bin/env python3
"""Axiom Full Pipeline Simulation — WebSocket server that replays the 3-min demo.

No Muse S needed. Streams synthetic EEG through the full Axiom pipeline:
  BrainStateEngine → AxiomAgent → (optional) LLM self-improvement

Connect the React frontend to ws://127.0.0.1:8080 to see everything live.
"""

import asyncio
import json
import time
import sys
import numpy as np
import websockets

from brain_state import BrainStateEngine
from axiom_agent import AxiomAgent, ScreenContext
from features import extract_state
from overlay_ctrl import OverlayController

EEG_SR = 256
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

clients: set = set()
engine = BrainStateEngine()
agent = AxiomAgent()
overlay = OverlayController()
USE_OVERLAY = True

np.random.seed(42)


DEMO_SCRIPT = [
    # (time_s, duration_s, eeg_state, clench, screen_context, description)
    (0, 3, "relaxed", False,
     {"active_app": "Desktop", "gaze_region": "center"},
     "Presence indicator wakes up"),
    (3, 2, "scanning", False,
     {"active_app": "Desktop", "gaze_region": "center"},
     "Calibration running"),
    (5, 2, "scanning", False,
     {"active_app": "Desktop", "gaze_target": "Safari", "gaze_region": "dock",
      "element_type": "app_icon", "available_actions": ["open"]},
     "Look at Safari in dock"),
    (7, 3, "focused", False,
     {"active_app": "Desktop", "gaze_target": "Safari", "gaze_region": "dock",
      "element_type": "app_icon"},
     "Engagement sustained on Safari"),
    (10, 1, "focused", True,
     {"active_app": "Desktop", "gaze_target": "Safari", "gaze_region": "dock"},
     "Jaw clench → Safari opens"),
    (11, 2, "focused", False,
     {"active_app": "Safari", "gaze_target": "URL Bar", "gaze_region": "top",
      "element_type": "text_field"},
     "Look at URL bar"),
    (13, 1, "focused", True,
     {"active_app": "Safari", "gaze_target": "gmail.com", "element_type": "link"},
     "Jaw clench → Gmail loads"),
    (14, 8, "reading", False,
     {"active_app": "Gmail", "gaze_target": "Email from Josh",
      "element_type": "email", "visible_text": "Hey, can we push the demo to Friday?"},
     "Reading email — high engagement"),
    (22, 3, "scanning", False,
     {"active_app": "Gmail",
      "available_actions": ["Reply", "Archive", "Star", "Back"]},
     "Engagement declining — ghost actions"),
    (25, 2, "focused", False,
     {"active_app": "Gmail", "gaze_target": "Reply", "element_type": "button",
      "available_actions": ["Reply", "Archive", "Star"]},
     "Gaze on Reply — intent ring"),
    (27, 1, "focused", True,
     {"active_app": "Gmail", "gaze_target": "Reply", "element_type": "button"},
     "Jaw clench → Reply draft opens"),
    (28, 5, "reading", False,
     {"active_app": "Gmail", "gaze_target": "Draft text",
      "element_type": "text_field",
      "visible_text": "Sounds good, Friday works. I'll prep by Thursday evening."},
     "Reading drafted reply"),
    (33, 1, "focused", True,
     {"active_app": "Gmail", "gaze_target": "Send", "element_type": "button"},
     "Jaw clench → Send email"),
    (34, 3, "relaxed", False,
     {"active_app": "Gmail", "gaze_region": "center"},
     "Engagement drops — context switch"),
    (37, 2, "scanning", False,
     {"active_app": "App Switcher", "gaze_target": "Spotify",
      "element_type": "app_icon",
      "available_actions": ["Gmail", "Safari", "Spotify", "VS Code"]},
     "App switcher — look at Spotify"),
    (39, 1, "focused", True,
     {"active_app": "App Switcher", "gaze_target": "Spotify",
      "element_type": "app_icon"},
     "Jaw clench → Spotify opens"),
    (40, 5, "focused", False,
     {"active_app": "Spotify", "gaze_target": "VS Code", "gaze_region": "dock",
      "element_type": "app_icon"},
     "Look at VS Code — intent ring"),
    (45, 3, "focused", False,
     {"active_app": "VS Code", "gaze_target": "Terminal", "element_type": "panel"},
     "VS Code — look at terminal"),
    (48, 5, "focused", False,
     {"active_app": "VS Code", "gaze_target": "Terminal",
      "element_type": "text_field"},
     "Deep focus — coding"),
    (53, 5, "relaxed", False,
     {"active_app": "VS Code", "gaze_region": "center"},
     "Demo complete — relaxing"),
]


def gen_eeg(seconds, state="normal", clench=False):
    n = int(EEG_SR * seconds)
    t = np.arange(n) / EEG_SR
    eeg = np.zeros((4, n))

    for ch in range(4):
        d = 8 * np.sin(2 * np.pi * 2.5 * t + ch * 0.5)
        th = 6 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        a = 12 * np.sin(2 * np.pi * 10 * t + ch * 0.7)
        b = 4 * np.sin(2 * np.pi * 22 * t + ch * 0.2)
        g = 2 * np.sin(2 * np.pi * 40 * t + ch * 0.9)
        noise = np.random.randn(n) * 3

        if state == "focused":
            b *= 3.0; g *= 2.5; a *= 0.5
        elif state == "relaxed":
            a *= 3.0; b *= 0.4
        elif state == "scanning":
            b *= 1.5; a *= 0.8
        elif state == "reading":
            b *= 2.0; g *= 1.5; th *= 1.5

        eeg[ch] = d + th + a + b + g + noise
        if ch in (0, 3):
            eeg[ch] *= 15

    if clench:
        for ch in (0, 3):
            eeg[ch, -25:] = np.random.randn(25) * 1000

    return eeg


async def broadcast(msg):
    dead = set()
    for ws in list(clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    for d in dead:
        clients.discard(d)


async def simulation_loop():
    """Run the demo script, streaming data at realistic rates."""
    prev_bands = None

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          AXIOM SIMULATION SERVER                        ║")
    print("║     Connect frontend to ws://127.0.0.1:8080             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Wait for first client
    while not clients:
        await asyncio.sleep(0.5)
    print("Client connected — starting demo simulation\n")

    for start_s, dur_s, eeg_state, clench, ctx_dict, description in DEMO_SCRIPT:
        print(f"  [{start_s:3d}s] {description}")

        ctx = ScreenContext(**{k: v for k, v in ctx_dict.items()
                               if hasattr(ScreenContext, k)})

        # Generate full EEG segment
        eeg = gen_eeg(max(1.0, dur_s), state=eeg_state, clench=clench)
        n_total = eeg.shape[1]

        # Stream in chunks at ~20Hz (50ms intervals)
        chunk_size = EEG_SR // 20  # ~13 samples per chunk
        t_sim = float(start_s)
        i = 0

        while i < n_total:
            end = min(i + chunk_size, n_total)
            chunk = eeg[:, i:end]
            n = chunk.shape[1]

            # Downsample for display
            step = max(1, n // 4)
            await broadcast(json.dumps({
                "type": "eeg",
                "channels": CHANNELS,
                "data": chunk[:, ::step].tolist(),
                "bands": {},
                "sampleRate": EEG_SR // step,
                "totalSamples": int(t_sim * EEG_SR),
                "timestamp": time.time(),
            }))

            # Brain state every ~0.5s
            if i % (EEG_SR // 2) < chunk_size:
                window = eeg[:, max(0, end - EEG_SR):end]
                if window.shape[1] >= EEG_SR:
                    brain = engine.process(window, timestamp=t_sim)
                    action = agent.decide(brain, ctx, timestamp=t_sim)

                    # RL state
                    state_seg = eeg[:, max(0, end - EEG_SR * 2):end]
                    if state_seg.shape[1] >= EEG_SR:
                        result = extract_state(state_seg, prev_bands)
                        prev_bands = result["bands"]
                        await broadcast(json.dumps({
                            "type": "state", **result,
                            "timestamp": time.time(),
                            "totalSamples": int(t_sim * EEG_SR),
                        }))

                    await broadcast(json.dumps({
                        "type": "brain",
                        "engagement": brain.engagement,
                        "focus": brain.focus,
                        "relaxation": brain.relaxation,
                        "cognitive_load": brain.cognitive_load,
                        "valence": brain.valence,
                        "jaw_clench": brain.jaw_clench,
                        "double_clench": brain.double_clench,
                        "context_switch": brain.context_switch,
                        "error_response": brain.error_response,
                        "timestamp": time.time(),
                    }))

                    # Update overlay
                    if USE_OVERLAY and overlay.running:
                        overlay.presence(True, brain.engagement)
                        overlay.mode(agent._current_mode)

                    if action.action_type != "none":
                        await broadcast(json.dumps({
                            "type": "action",
                            "action_type": action.action_type,
                            "target": action.target,
                            "confidence": action.confidence,
                            "reason": action.reason,
                            "timestamp": time.time(),
                        }))

                        if USE_OVERLAY and overlay.running:
                            if action.action_type in ("click", "open"):
                                overlay.flash("green")
                            elif action.action_type == "undo":
                                overlay.flash("red")
                            elif "intent_ring" in action.action_type:
                                overlay.intent_ring(True, action.confidence)

            t_sim += n / EEG_SR
            i = end
            await asyncio.sleep(0.05)

        # Learning metrics after each step
        await broadcast(json.dumps({
            "type": "learning",
            "accuracy": agent.accuracy,
            "accuracy_history": agent.accuracy_history[-100:],
            "total_actions": agent.total_actions,
            "undone_actions": agent.undone_actions,
            "thresholds": agent.thresholds,
            "decision_log": agent.get_decision_log(10),
            "mode": agent._current_mode,
            "description": description,
            "timestamp": time.time(),
        }))

    print("\n  DEMO COMPLETE")
    print(f"  Actions: {agent.total_actions}, Accuracy: {agent.accuracy:.0%}")
    print(f"  Mode: {agent._current_mode}")

    # Loop the demo
    print("\n  Restarting demo in 5s...")
    await asyncio.sleep(5)
    asyncio.create_task(simulation_loop())


async def handler(ws):
    clients.add(ws)
    print(f"Client connected ({len(clients)} total)", flush=True)
    try:
        await ws.send(json.dumps({
            "type": "info",
            "channels": CHANNELS,
            "sampleRate": EEG_SR,
            "bands": ["delta", "theta", "alpha", "beta", "gamma"],
            "stateSize": 29,
            "features": ["brain_state", "agent", "learning"],
            "simulation": True,
        }))
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)
        print(f"Client disconnected ({len(clients)} total)", flush=True)


async def main():
    global USE_OVERLAY

    if "--no-overlay" in sys.argv:
        USE_OVERLAY = False

    if USE_OVERLAY:
        try:
            overlay.start()
            overlay.presence(True, 0.3)
            overlay.mode("calibrating")
            overlay.status("AXIOM SIMULATION")
        except Exception as e:
            print(f"  [OVERLAY] Failed: {e}")
            USE_OVERLAY = False

    server = await websockets.serve(handler, "127.0.0.1", 8080)
    print("Axiom simulation server on ws://127.0.0.1:8080", flush=True)
    print(f"  Overlay: {'ON' if USE_OVERLAY else 'OFF'}", flush=True)
    asyncio.create_task(simulation_loop())
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
