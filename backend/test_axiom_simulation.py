#!/usr/bin/env python3
"""Full Axiom simulation: runs through the 3-minute demo script from the spec.

Uses synthetic EEG data based on real Muse S recording stats.
Simulates gaze positions and screen context.
"""

import time
import numpy as np
from brain_state import BrainStateEngine
from axiom_agent import AxiomAgent, ScreenContext

np.random.seed(42)
EEG_SR = 256


def gen_eeg(seconds: float, state: str = "normal", clench: bool = False) -> np.ndarray:
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


def simulate_demo():
    engine = BrainStateEngine()
    agent = AxiomAgent()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          AXIOM — FULL DEMO SIMULATION                   ║")
    print("║     3-minute demo script from product spec              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    steps = [
        # (time_label, duration, eeg_state, clench, screen_context, description)
        ("0:00", 0.5, "relaxed", False,
         ScreenContext(active_app="Desktop", gaze_target="", gaze_region="center"),
         "Put on headband. Presence indicator wakes up."),

        ("0:05", 0.5, "scanning", False,
         ScreenContext(active_app="Desktop", gaze_target="", gaze_region="center"),
         "Calibration running (eye tracking + alpha baseline)."),

        ("0:30", 0.5, "scanning", False,
         ScreenContext(active_app="Desktop", gaze_target="Safari", gaze_region="dock",
                       element_type="app_icon", available_actions=["open"]),
         "Look at browser in dock. Intent ring starts."),

        ("0:32", 1.5, "focused", False,
         ScreenContext(active_app="Desktop", gaze_target="Safari", gaze_region="dock",
                       element_type="app_icon", available_actions=["open"]),
         "Engagement sustained on Safari. Intent ring filling..."),

        ("0:34", 0.5, "focused", True,
         ScreenContext(active_app="Desktop", gaze_target="Safari", gaze_region="dock",
                       element_type="app_icon"),
         "Jaw clench! Safari opens instantly."),

        ("0:35", 0.5, "focused", False,
         ScreenContext(active_app="Safari", gaze_target="URL Bar", gaze_region="top",
                       element_type="text_field"),
         "Look at URL bar. Axiom suggests gmail.com."),

        ("0:37", 0.5, "focused", True,
         ScreenContext(active_app="Safari", gaze_target="gmail.com suggestion",
                       element_type="link"),
         "Jaw clench on suggestion. Gmail loads."),

        ("1:00", 2.0, "reading", False,
         ScreenContext(active_app="Gmail", gaze_target="Email from Josh",
                       element_type="email", visible_text="Hey, can we push the demo to Friday?"),
         "Reading email. Engagement tracked. High focus."),

        ("1:20", 0.5, "scanning", False,
         ScreenContext(active_app="Gmail", gaze_target="",
                       available_actions=["Reply", "Archive", "Star", "Back"]),
         "Engagement declining. Ghost actions appear."),

        ("1:22", 0.5, "focused", False,
         ScreenContext(active_app="Gmail", gaze_target="Reply",
                       element_type="button",
                       available_actions=["Reply", "Archive", "Star"]),
         "Gaze on Reply. Intent ring starts."),

        ("1:23", 0.5, "focused", True,
         ScreenContext(active_app="Gmail", gaze_target="Reply", element_type="button"),
         "Jaw clench! Reply draft opens."),

        ("1:25", 1.0, "reading", False,
         ScreenContext(active_app="Gmail", gaze_target="Draft text",
                       element_type="text_field",
                       visible_text="Sounds good, Friday works for me. I'll prep the demo by Thursday evening."),
         "Axiom drafted reply from context. User reads it. Engagement holds = good."),

        ("1:35", 0.5, "focused", True,
         ScreenContext(active_app="Gmail", gaze_target="Send", element_type="button"),
         "Look at Send. Jaw clench. Email sent."),

        ("1:40", 0.5, "relaxed", False,
         ScreenContext(active_app="Gmail", gaze_target="", gaze_region="center"),
         "Engagement drops. Context switch pattern detected."),

        ("1:45", 0.5, "scanning", False,
         ScreenContext(active_app="App Switcher", gaze_target="Spotify",
                       element_type="app_icon",
                       available_actions=["Gmail", "Safari", "Spotify", "VS Code"]),
         "4-app overlay appears. Look at Spotify."),

        ("1:47", 0.5, "focused", True,
         ScreenContext(active_app="App Switcher", gaze_target="Spotify",
                       element_type="app_icon"),
         "Jaw clench. Spotify opens. Music starts."),

        ("2:00", 0.5, "scanning", False,
         ScreenContext(active_app="Spotify", gaze_target="VS Code", gaze_region="dock",
                       element_type="app_icon"),
         "Look at VS Code in dock. Intent ring starts."),

        ("2:02", 1.0, "focused", False,
         ScreenContext(active_app="Spotify", gaze_target="VS Code", gaze_region="dock",
                       element_type="app_icon"),
         "Engagement sustained. Intent ring fills. VS Code opens."),

        ("2:10", 0.5, "focused", False,
         ScreenContext(active_app="VS Code", gaze_target="Terminal",
                       element_type="panel"),
         "Look at terminal. Axiom opens it."),

        ("2:30", 0.5, "focused", False,
         ScreenContext(active_app="VS Code", gaze_target="Terminal",
                       element_type="text_field"),
         "Focus spikes. Axiom runs last command (npm start)."),

        ("3:00", 0.5, "relaxed", False,
         ScreenContext(active_app="VS Code", gaze_target="", gaze_region="center"),
         "Demo complete. Show dashboard. Accuracy curve climbing."),
    ]

    t_sim = 0.0
    print()
    for time_label, duration, eeg_state, clench, ctx, description in steps:
        eeg = gen_eeg(max(1.0, duration), state=eeg_state, clench=clench)
        brain = engine.process(eeg, timestamp=t_sim)
        action = agent.decide(brain, ctx, timestamp=t_sim)

        # Color-code action types
        action_display = action.action_type.upper()
        if action.action_type in ("click", "open"):
            action_display = f"\033[32m{action_display}\033[0m"  # green
        elif action.action_type == "undo":
            action_display = f"\033[31m{action_display}\033[0m"  # red
        elif "intent_ring" in action.action_type:
            action_display = f"\033[33m{action_display}\033[0m"  # yellow
        elif action.action_type == "ghost_actions":
            action_display = f"\033[36m{action_display}\033[0m"  # cyan
        elif action.action_type == "none":
            action_display = f"\033[90m{action_display}\033[0m"  # gray

        print(f"  [{time_label}] {description}")
        print(f"          Brain: eng={brain.engagement:.2f} foc={brain.focus:.2f} "
              f"rel={brain.relaxation:.2f} clench={brain.jaw_clench}")
        print(f"          Agent: {action_display} → {action.target or '—'}  "
              f"(conf={action.confidence:.2f}, {action.reason})")
        print()

        t_sim += duration

    # Final stats
    print("─" * 60)
    print(f"  DEMO COMPLETE")
    print(f"  Total actions:  {agent.total_actions}")
    print(f"  Accuracy:       {agent.accuracy:.1%}")
    print(f"  Undone:         {agent.undone_actions}")
    print(f"  Mode:           {agent._current_mode}")
    print(f"  Thresholds:     {json.dumps(agent.thresholds, indent=4)}")
    print("─" * 60)

    # Decision log
    print("\n  DECISION LOG (last 10):")
    for entry in agent.get_decision_log(10):
        print(f"    [{entry['action_type']:20s}] target={entry['target']:20s} "
              f"conf={entry['confidence']:.2f}  {entry['reason'][:50]}")


import json
if __name__ == "__main__":
    simulate_demo()
