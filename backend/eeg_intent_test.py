#!/usr/bin/env python3
"""Axiom EEG Intent Test — can we detect browser task intent from brain signals?

Presents browser-like tasks, records timestamped EEG, saves everything
to disk, then runs post-analysis with proper mental-state metrics.

Tasks simulate real browser intents:
  - Idle/waiting (baseline)
  - Scanning (visual search among elements)
  - Reading (sustained attention on text)
  - Deciding (choosing between options)
  - Pre-click intent (about to act)
  - Comparing (analytical evaluation)

Saves raw data to backend/data/intent_session_<timestamp>.npz
"""

import cv2
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, field
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)
from muse_bridge import EEGBridgeClient, is_bridge_running

EEG_SR = 256
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Task:
    name: str
    category: str  # idle, scan, read, decide, preclick, compare
    instruction: str
    duration: int
    draw_fn: str  # name of drawing function


@dataclass
class Recording:
    task_name: str
    category: str
    start_time: float = 0.0
    end_time: float = 0.0
    click_times: list = field(default_factory=list)
    eeg_samples: list = field(default_factory=lambda: [[] for _ in range(4)])


TASKS = [
    # 1. Baseline idle
    Task("IDLE - WAIT", "idle",
         "Stare at the dot. Wait for the next task.", 10, "draw_idle"),

    # 2. Visual search — find the target among distractors
    Task("SCAN - FIND TARGET", "scan",
         "Find the GREEN circle among the shapes. Click SPACE when you find it.", 12, "draw_scan"),

    # 3. Reading — sustained attention
    Task("READ - COMPREHEND", "read",
         "Read the paragraph carefully. You'll answer a question after.", 15, "draw_read"),

    # 4. Decision — choose between options
    Task("DECIDE - PICK ONE", "decide",
         "Three options below. Think about which you'd pick. Press 1, 2, or 3.", 12, "draw_decide"),

    # 5. Pre-click — intent to act, waiting for the right moment
    Task("PRE-CLICK - WAIT FOR GREEN", "preclick",
         "A circle will turn GREEN. Press SPACE the instant it does.", 15, "draw_preclick"),

    # 6. Idle again (for comparison)
    Task("IDLE - REST", "idle",
         "Close your eyes briefly, then stare at the dot.", 10, "draw_idle"),

    # 7. Comparing — analytical
    Task("COMPARE - WHICH IS MORE?", "compare",
         "Compare the two data cards. Which service is better value? Press 1 or 2.", 12, "draw_compare"),

    # 8. Scan again — different layout
    Task("SCAN - FIND WORD", "scan",
         "Find the word 'AXIOM' hidden in the grid. Press SPACE when found.", 12, "draw_scan2"),

    # 9. Pre-click again
    Task("PRE-CLICK - COUNTDOWN", "preclick",
         "Watch the countdown. Press SPACE right when it hits zero.", 15, "draw_preclick2"),

    # 10. Reading again
    Task("READ - TECHNICAL", "read",
         "Read this technical passage carefully.", 15, "draw_read2"),
]


# ═══════════════════════════════════════════════════════════
# Drawing functions for each task type
# ═══════════════════════════════════════════════════════════

def draw_idle(canvas, elapsed, sw, sh, state):
    mid_x, mid_y = sw // 4, sh // 2
    pulse = int(4 + 2 * np.sin(elapsed * 2))
    cv2.circle(canvas, (mid_x, mid_y), pulse, (0, 200, 100), -1)


def draw_scan(canvas, elapsed, sw, sh, state):
    if "shapes" not in state:
        rng = np.random.RandomState(42)
        shapes = []
        target_idx = rng.randint(15, 30)
        for i in range(35):
            x = 40 + rng.randint(0, sw // 2 - 100)
            y = 120 + rng.randint(0, sh - 200)
            is_target = (i == target_idx)
            # Pre-generate shape type and color so they don't flicker
            shape_type = "circle" if rng.random() > 0.5 else "rect"
            color = (0, 0, 200) if rng.random() > 0.5 else (200, 200, 0)
            shapes.append((x, y, is_target, shape_type, color))
        state["shapes"] = shapes

    for x, y, is_target, shape_type, color in state["shapes"]:
        if is_target:
            # Target is green but same shape as others — harder to find
            cv2.circle(canvas, (x, y), 14, (0, 180, 0), -1)
        else:
            if shape_type == "circle":
                cv2.circle(canvas, (x, y), 14, color, -1)
            else:
                cv2.rectangle(canvas, (x - 12, y - 12), (x + 12, y + 12), color, -1)
    if not state.get("found"):
        cv2.putText(canvas, "Press SPACE when you find the green circle", (40, sh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)


def draw_read(canvas, elapsed, sw, sh, state):
    text = [
        "Neural interfaces are transforming how humans interact with",
        "technology. By decoding brain signals in real time, these systems",
        "can predict user intent before any physical action occurs. The",
        "primary challenge lies in separating meaningful neural patterns",
        "from biological noise — muscle artifacts, eye blinks, and",
        "environmental interference all contaminate the signal. Modern",
        "approaches use a combination of spatial filtering, spectral",
        "analysis, and machine learning to extract the cognitive state",
        "information buried within the raw electroencephalogram. With",
        "consumer devices like the Muse S, researchers have shown that",
        "basic mental states can be classified with 70-85% accuracy.",
    ]
    for i, line in enumerate(text):
        cv2.putText(canvas, line, (50, 140 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 230), 1)


def draw_decide(canvas, elapsed, sw, sh, state):
    options = [
        ("1. STREAM", "Watch a movie on Netflix", "Relaxing, passive"),
        ("2. BUILD", "Code a new project feature", "Active, creative"),
        ("3. LEARN", "Take an online course lesson", "Focused, absorbing"),
    ]
    for i, (title, desc, mood) in enumerate(options):
        bx = 50
        by = 160 + i * 140
        bw = sw // 2 - 100
        bh = 120
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (30, 30, 60), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 120), 1)
        cv2.putText(canvas, title, (bx + 16, by + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
        cv2.putText(canvas, desc, (bx + 16, by + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1)
        cv2.putText(canvas, mood, (bx + 16, by + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1)


def draw_preclick(canvas, elapsed, sw, sh, state):
    mid_x, mid_y = sw // 4, sh // 2
    if "trigger_time" not in state:
        state["trigger_time"] = 6.0 + (hash("preclick1") % 30) / 10.0  # 6-9s deterministic
    if elapsed < state["trigger_time"]:
        cv2.circle(canvas, (mid_x, mid_y), 40, (0, 0, 180), -1)
        cv2.putText(canvas, "WAIT...", (mid_x - 40, mid_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
        cv2.putText(canvas, "Stare at the circle. Press SPACE when it turns GREEN.",
                    (40, sh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1)
    else:
        pulse = int(40 + 5 * np.sin(elapsed * 8))
        cv2.circle(canvas, (mid_x, mid_y), pulse, (0, 255, 0), -1)
        cv2.putText(canvas, "NOW! PRESS SPACE!", (mid_x - 100, mid_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_preclick2(canvas, elapsed, sw, sh, state):
    mid_x, mid_y = sw // 4, sh // 2
    countdown = max(0, 7 - elapsed)
    if countdown > 0:
        # Red background countdown
        cv2.circle(canvas, (mid_x, mid_y), 60, (0, 0, 100), -1)
        cv2.putText(canvas, f"{countdown:.1f}", (mid_x - 40, mid_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        cv2.putText(canvas, "Get ready... press SPACE at ZERO",
                    (40, sh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1)
    else:
        pulse = int(50 + 8 * np.sin(elapsed * 8))
        cv2.circle(canvas, (mid_x, mid_y), pulse, (0, 255, 0), -1)
        cv2.putText(canvas, "GO!", (mid_x - 30, mid_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(canvas, "PRESS SPACE NOW!", (mid_x - 100, mid_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_compare(canvas, elapsed, sw, sh, state):
    cards = [
        ("Plan A: BASIC", "$9.99/mo", "10 GB storage", "Email support", "3 projects"),
        ("Plan B: PRO", "$19.99/mo", "100 GB storage", "Priority support", "Unlimited projects"),
    ]
    for i, (title, price, f1, f2, f3) in enumerate(cards):
        bx = 50 + i * (sw // 4)
        by = 160
        bw = sw // 4 - 60
        bh = 300
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (25, 25, 50), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 120), 2)
        cv2.putText(canvas, title, (bx + 16, by + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)
        cv2.putText(canvas, price, (bx + 16, by + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
        for fi, feat in enumerate([f1, f2, f3]):
            cv2.putText(canvas, f"  {feat}", (bx + 16, by + 130 + fi * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 190), 1)
        cv2.putText(canvas, f"Press {i+1} to choose", (bx + 16, by + bh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1)


def draw_scan2(canvas, elapsed, sw, sh, state):
    if "words" not in state:
        rng = np.random.RandomState(99)
        # All similar-looking words, same color — AXIOM blends in
        words_pool = ["BRAIN", "SIGNAL", "NEURAL", "FOCUS", "ALPHA", "THETA",
                       "GAMMA", "DELTA", "PULSE", "WAVE", "AXION", "AXIOM",
                       "CORTEX", "SYNTH", "AXIAL", "PRISM", "NEXUS", "ATLAS"]
        grid = []
        for r in range(8):
            for c in range(6):
                w = rng.choice(words_pool)
                grid.append((40 + c * (sw // 12), 130 + r * 45, w))
        # Place exactly one AXIOM in a random spot
        idx = rng.randint(20, 40)
        grid[idx] = (grid[idx][0], grid[idx][1], "AXIOM")
        state["words"] = grid

    # All words same color — must actually read each one
    for x, y, word in state["words"]:
        cv2.putText(canvas, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 170), 1)
    if not state.get("found"):
        cv2.putText(canvas, "Find 'AXIOM' among the words. Press SPACE when found.",
                    (40, sh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)


def draw_read2(canvas, elapsed, sw, sh, state):
    text = [
        "The engagement index, defined as beta power divided by the sum",
        "of alpha and theta power, has been validated as a reliable",
        "marker of cognitive workload across multiple studies. When a",
        "user transitions from passive browsing to active decision-",
        "making, the engagement index typically increases by 15-40%.",
        "Frontal alpha asymmetry (FAA), computed as the log difference",
        "between right and left frontal alpha, reflects approach vs.",
        "withdrawal motivation. Positive FAA suggests the user is",
        "inclined to engage or act, while negative FAA suggests",
        "avoidance or disinterest. These two metrics together can",
        "approximate whether a user intends to interact with content.",
    ]
    for i, line in enumerate(text):
        cv2.putText(canvas, line, (50, 140 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 230), 1)


DRAW_FNS = {
    "draw_idle": draw_idle, "draw_scan": draw_scan, "draw_read": draw_read,
    "draw_decide": draw_decide, "draw_preclick": draw_preclick,
    "draw_preclick2": draw_preclick2, "draw_compare": draw_compare,
    "draw_scan2": draw_scan2, "draw_read2": draw_read2,
}


# ═══════════════════════════════════════════════════════════
# EEG Processing Utilities
# ═══════════════════════════════════════════════════════════

def filter_channel(data):
    if len(data) < 12:
        return data.copy()
    out = data.copy()
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
    return out


def get_band_powers_per_channel(data_1ch):
    if len(data_1ch) < EEG_SR:
        return {name: 0.0 for name, _, _ in BANDS}
    filtered = filter_channel(data_1ch)
    nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
    psd = DataFilter.get_psd_welch(filtered, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
    return {name: float(DataFilter.get_band_power(psd, lo, hi)) for name, lo, hi in BANDS}


def draw_waveform(canvas, data, x, y, w, h, color, label="", scale=80):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 50), 1)
    if label:
        cv2.putText(canvas, label, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    n = len(data)
    if n < 2:
        return
    show = min(n, w)
    segment = data[-show:]
    mid_y = y + h // 2
    pts = []
    for i, val in enumerate(segment):
        px = x + int(i * w / show)
        py = mid_y - int(np.clip(val / scale, -1, 1) * (h // 2 - 4))
        pts.append((px, py))
    if len(pts) > 1:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════

def run_experiment():
    from screeninfo import get_monitors
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1470, 956

    print(f"Screen: {sw}x{sh}")

    if not is_bridge_running():
        print("ERROR: Start muse_bridge.py first")
        sys.exit(1)

    eeg = EEGBridgeClient()
    eeg.start()
    print("Connected to Muse bridge")

    print("Warming up...")
    for _ in range(40):
        eeg.pull()
        time.sleep(0.05)

    cv2.namedWindow("EEG Intent Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("EEG Intent Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recordings = []
    task_idx = 0

    print(f"\n{len(TASKS)} tasks. SPACE to start each. Keep window focused!\n")

    while task_idx < len(TASKS):
        task = TASKS[task_idx]
        phase = "ready"
        rec_start = 0.0
        rec = Recording(task_name=task.name, category=task.category)
        draw_state = {}

        while True:
            new_data = eeg.pull()
            window = eeg.get_window(2.0)

            if phase == "recording" and new_data.shape[1] > 0:
                for ch in range(4):
                    rec.eeg_samples[ch].extend(new_data[ch].tolist())

            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            canvas[:] = (20, 6, 6)

            elapsed = time.time() - rec_start if phase == "recording" else 0

            if phase == "ready":
                cv2.putText(canvas, f"TASK {task_idx+1}/{len(TASKS)}", (40, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 104, 148), 1)
                cat_colors = {"idle": (128,128,128), "scan": (255,200,0), "read": (200,200,100),
                              "decide": (0,200,255), "preclick": (0,255,136), "compare": (255,100,200)}
                cc = cat_colors.get(task.category, (128,128,128))
                cv2.putText(canvas, task.name, (sw // 4 - 200, sh // 2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, cc, 2)
                cv2.putText(canvas, "Press SPACE to start", (sw // 4 - 130, sh // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                cv2.putText(canvas, task.instruction, (sw // 4 - 250, sh // 2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)

            elif phase == "recording":
                remaining = max(0, task.duration - elapsed)
                progress = elapsed / task.duration
                cv2.putText(canvas, task.name, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)
                cv2.putText(canvas, f"{remaining:.0f}s", (sw // 2 - 30, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
                bar_x, bar_w = 40, sw // 2 - 80
                cv2.rectangle(canvas, (bar_x, 50), (bar_x + bar_w, 55), (40, 40, 60), -1)
                cv2.rectangle(canvas, (bar_x, 50), (bar_x + int(bar_w * progress), 55), (0, 255, 136), -1)
                pulse = int(128 + 127 * np.sin(elapsed * 4))
                cv2.circle(canvas, (sw // 2 + 20, 30), 5, (0, 0, pulse), -1)

                # Draw task visuals
                draw_fn = DRAW_FNS.get(task.draw_fn)
                if draw_fn:
                    draw_fn(canvas, elapsed, sw, sh, draw_state)

                cv2.putText(canvas, task.instruction, (40, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 160), 1)

                if elapsed >= task.duration:
                    rec.end_time = time.time()
                    recordings.append(rec)
                    n = len(rec.eeg_samples[0])
                    print(f"  [{task.name}] {n} samples ({n/EEG_SR:.1f}s), "
                          f"{len(rec.click_times)} clicks")
                    task_idx += 1
                    time.sleep(0.3)
                    break

            # Live EEG (right panel)
            px = int(sw * 0.55)
            pw = sw - px - 15
            cv2.line(canvas, (px - 10, 0), (px - 10, sh), (30, 30, 50), 1)
            cv2.putText(canvas, "LIVE EEG", (px, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)

            if window.shape[1] >= EEG_SR:
                wf_h = (sh - 100) // 4
                ch_colors = [(0,200,200), (200,100,255), (255,100,200), (200,200,0)]
                for ci in range(4):
                    wy = 45 + ci * wf_h
                    filtered = filter_channel(window[ci])
                    draw_waveform(canvas, filtered, px, wy, pw, wf_h - 4,
                                  ch_colors[ci], label=CH_NAMES[ci])

            cv2.imshow("EEG Intent Test", canvas)
            key = cv2.waitKey(16) & 0xFF

            if key == 27:
                print("Aborted.")
                break
            elif key == 32:  # SPACE
                if phase == "ready":
                    phase = "recording"
                    rec_start = time.time()
                    rec.start_time = rec_start
                    rec.eeg_samples = [[] for _ in range(4)]
                    draw_state = {}
                    print(f"  [{task.name}] Recording...")
                elif phase == "recording":
                    rec.click_times.append(time.time() - rec_start)
                    # For scan/preclick, SPACE = "I found it / I clicked" — end task
                    if task.category in ("scan", "preclick"):
                        draw_state["found"] = True
                        # Record 1 more second of post-action EEG then end
                        if elapsed > 2:  # must have been searching for at least 2s
                            task_duration_override = elapsed + 1.0
                            if not hasattr(task, '_override'):
                                task._override = True
                                task.duration = int(task_duration_override) + 1
            elif key in [ord('1'), ord('2'), ord('3')] and phase == "recording":
                rec.click_times.append(time.time() - rec_start)
                # For decide/compare, picking an option ends the task
                if task.category in ("decide", "compare"):
                    choice = key - ord('0')
                    print(f"    Chose option {choice}")
                    if elapsed > 2:
                        task_duration_override = elapsed + 1.0
                        if not hasattr(task, '_override'):
                            task._override = True
                            task.duration = int(task_duration_override) + 1

        if key == 27:
            break

    cv2.destroyAllWindows()
    eeg.stop()

    if not recordings:
        print("No data recorded.")
        return

    # ═══════════════════════════════════════════════════════
    # Save raw data
    # ═══════════════════════════════════════════════════════
    os.makedirs(DATA_DIR, exist_ok=True)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(DATA_DIR, f"intent_session_{session_id}.npz")

    save_data = {"session_id": session_id, "sample_rate": EEG_SR, "n_tasks": len(recordings)}
    for i, rec in enumerate(recordings):
        eeg_arr = np.array([np.array(ch, dtype=np.float64) for ch in rec.eeg_samples])
        save_data[f"task_{i}_name"] = rec.task_name
        save_data[f"task_{i}_category"] = rec.category
        save_data[f"task_{i}_start"] = rec.start_time
        save_data[f"task_{i}_end"] = rec.end_time
        save_data[f"task_{i}_clicks"] = np.array(rec.click_times)
        save_data[f"task_{i}_eeg"] = eeg_arr

    np.savez(save_path, **save_data)
    print(f"\nRaw data saved to: {save_path}")
    print(f"  {len(recordings)} tasks, session {session_id}")

    # ═══════════════════════════════════════════════════════
    # Post-analysis
    # ═══════════════════════════════════════════════════════
    run_analysis(recordings, save_path)


def run_analysis(recordings, save_path):
    """Compute proper mental-state metrics with artifact rejection."""
    print("\n" + "=" * 100)
    print("POST-ANALYSIS — MENTAL STATE METRICS")
    print("=" * 100)

    ARTIFACT_THRESHOLD = 500.0  # Muse S raw units — reject only extreme spikes

    all_results = []

    for rec in recordings:
        eeg = np.array([np.array(ch, dtype=np.float64) for ch in rec.eeg_samples])
        if eeg.shape[1] < EEG_SR * 2:
            print(f"  [{rec.task_name}] Too short, skipping")
            continue

        # Sliding window analysis: 2s windows, 1s step
        window_samples = EEG_SR * 2
        step_samples = EEG_SR
        n_windows = max(1, (eeg.shape[1] - window_samples) // step_samples + 1)

        window_metrics = []
        clean_windows = 0
        rejected_windows = 0

        for wi in range(n_windows):
            start = wi * step_samples
            end = start + window_samples
            if end > eeg.shape[1]:
                break

            segment = eeg[:, start:end]

            # Artifact rejection: check if any channel exceeds threshold
            max_amplitude = np.max(np.abs(segment))
            max_gradient = np.max(np.abs(np.diff(segment, axis=1))) * EEG_SR / 1e6
            if max_amplitude > ARTIFACT_THRESHOLD:
                rejected_windows += 1
                continue

            clean_windows += 1

            # Compute band powers per channel
            ch_bands = {}
            for ci in range(4):
                ch_bands[ci] = get_band_powers_per_channel(segment[ci])

            # Average across channels
            avg = {name: np.mean([ch_bands[ci][name] for ci in range(4)]) for name, _, _ in BANDS}
            # Frontal only (AF7=1, AF8=2)
            frontal = {name: np.mean([ch_bands[1][name], ch_bands[2][name]]) for name, _, _ in BANDS}
            # Temporal only (TP9=0, TP10=3)
            temporal = {name: np.mean([ch_bands[0][name], ch_bands[3][name]]) for name, _, _ in BANDS}

            total = sum(avg.values()) + 0.001
            f_total = sum(frontal.values()) + 0.001

            # Derived metrics
            engagement = avg["beta"] / (avg["alpha"] + avg["theta"] + 0.001)
            theta_beta = avg["theta"] / (avg["beta"] + 0.001)
            alpha_theta = avg["alpha"] / (avg["theta"] + 0.001)

            # Frontal metrics
            f_engagement = frontal["beta"] / (frontal["alpha"] + frontal["theta"] + 0.001)
            # Frontal alpha asymmetry: ln(AF8) - ln(AF7)
            af7_alpha = ch_bands[1]["alpha"] + 0.001
            af8_alpha = ch_bands[2]["alpha"] + 0.001
            faa = float(np.log(af8_alpha) - np.log(af7_alpha))

            # Relative band powers
            rel = {name: avg[name] / total * 100 for name, _, _ in BANDS}
            f_rel = {name: frontal[name] / f_total * 100 for name, _, _ in BANDS}

            window_metrics.append({
                "engagement": engagement,
                "f_engagement": f_engagement,
                "theta_beta": theta_beta,
                "alpha_theta": alpha_theta,
                "faa": faa,
                "rel_alpha": rel["alpha"],
                "rel_beta": rel["beta"],
                "rel_theta": rel["theta"],
                "f_rel_alpha": f_rel["alpha"],
                "f_rel_beta": f_rel["beta"],
            })

        if not window_metrics:
            print(f"  [{rec.task_name}] All windows rejected as artifacts")
            continue

        # Average metrics across clean windows
        avg_metrics = {}
        for key in window_metrics[0].keys():
            vals = [w[key] for w in window_metrics]
            avg_metrics[key] = np.mean(vals)
            avg_metrics[f"{key}_std"] = np.std(vals)

        avg_metrics["clean_pct"] = clean_windows / (clean_windows + rejected_windows) * 100
        avg_metrics["n_clean"] = clean_windows
        avg_metrics["n_rejected"] = rejected_windows

        all_results.append((rec.task_name, rec.category, avg_metrics))

    # Print results
    print(f"\n{'TASK':<28s} {'Cat':>8s} {'Engage':>8s} {'F.Eng':>8s} {'θ/β':>7s} "
          f"{'FAA':>7s} {'α%':>6s} {'β%':>6s} {'Clean%':>7s}")
    print("-" * 100)

    for name, cat, m in all_results:
        print(f"{name:<28s} {cat:>8s} "
              f"{m['engagement']:>8.3f} {m['f_engagement']:>8.3f} "
              f"{m['theta_beta']:>7.2f} {m['faa']:>7.3f} "
              f"{m['rel_alpha']:>5.1f}% {m['rel_beta']:>5.1f}% "
              f"{m['clean_pct']:>6.0f}%")

    # Category averages
    print(f"\n{'CATEGORY AVERAGES':}")
    print("-" * 80)
    categories = {}
    for name, cat, m in all_results:
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    print(f"{'Category':<12s} {'Engage':>8s} {'F.Eng':>8s} {'θ/β':>7s} {'FAA':>7s} {'α%':>6s} {'β%':>6s}")
    print("-" * 60)
    for cat in ["idle", "scan", "read", "decide", "preclick", "compare"]:
        if cat not in categories:
            continue
        ms = categories[cat]
        print(f"{cat:<12s} "
              f"{np.mean([m['engagement'] for m in ms]):>8.3f} "
              f"{np.mean([m['f_engagement'] for m in ms]):>8.3f} "
              f"{np.mean([m['theta_beta'] for m in ms]):>7.2f} "
              f"{np.mean([m['faa'] for m in ms]):>7.3f} "
              f"{np.mean([m['rel_alpha'] for m in ms]):>5.1f}% "
              f"{np.mean([m['rel_beta'] for m in ms]):>5.1f}%")

    # Can we separate intents?
    print("\n" + "=" * 100)
    print("INTENT SEPARATION ANALYSIS")
    print("=" * 100)

    for metric_name, desc in [
        ("engagement", "Engagement β/(α+θ) — higher = more focused"),
        ("f_engagement", "Frontal Engagement — cognitive focus specifically"),
        ("theta_beta", "Theta/Beta — higher = less focused"),
        ("faa", "Frontal Alpha Asymmetry — positive = approach motivation"),
    ]:
        print(f"\n  {desc}:")
        vals = [(cat, np.mean([m[metric_name] for m in ms]))
                for cat, ms in categories.items()]
        vals.sort(key=lambda x: x[1])
        if not vals:
            print("    No data")
            continue
        min_v, max_v = vals[0][1], vals[-1][1]
        spread = max_v - min_v
        for cat, v in vals:
            bar_len = int((v - min_v) / (spread + 0.001) * 40)
            print(f"    {cat:<12s} {v:>8.3f} {'█' * bar_len}")
        print(f"    Spread: {spread:.3f} ({'GOOD' if spread > 0.03 else 'WEAK'} separation)")

    print(f"\nData saved at: {save_path}")
    print("Reload with: data = np.load('path.npz', allow_pickle=True)")


if __name__ == "__main__":
    run_experiment()
