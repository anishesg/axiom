#!/usr/bin/env python3
"""Axiom EEG Explorer — map real tasks to brain signals.

Connects to the Muse bridge, presents tasks one at a time,
records timestamped EEG throughout, then shows a full analysis
of how signals line up with each task.

Usage:
    python3 eeg_explore.py

Requires: muse_bridge.py running (real or --sim)
"""

import cv2
import numpy as np
import sys
import time
from dataclasses import dataclass, field
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes, WindowOperations

from muse_bridge import EEGBridgeClient, is_bridge_running

EEG_SR = 256
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
BANDS = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]
BAND_COLORS = {
    "delta": (200, 150, 50),
    "theta": (50, 200, 200),
    "alpha": (50, 200, 50),
    "beta": (50, 100, 255),
    "gamma": (200, 50, 200),
}


@dataclass
class Task:
    name: str
    instruction: str
    detail: str
    duration: int
    category: str  # "clench", "blink", "focus", "relax", "movement"


@dataclass
class TaskRecording:
    task: Task
    start_time: float = 0.0
    end_time: float = 0.0
    eeg_data: np.ndarray = field(default_factory=lambda: np.zeros((4, 0)))


READING_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "A brain-computer interface translates neural",
    "signals into commands. The Muse S headband has",
    "4 EEG channels sampling at 256 Hz. Alpha waves",
    "indicate relaxation while beta waves indicate",
    "active thinking and concentration. Gamma waves",
    "above 30 Hz are associated with peak cognition",
    "and cross-modal sensory processing in the brain.",
]

TASKS = [
    Task("BASELINE", "Sit still and stare at the dot",
         "Don't think about anything specific. Just exist.\nKeep your eyes on the green dot.",
         10, "baseline"),

    Task("JAW CLENCH x5", "Clench your jaw firmly 5 times",
         "Clench hard for 1 second, release for 1 second.\nRepeat 5 times. Creates huge EMG on TP9/TP10.",
         12, "clench"),

    Task("SLOW BLINKS x5", "Blink slowly and deliberately 5 times",
         "Close eyes fully for 1 second, open for 1 second.\nRepeat. Blinks spike AF7/AF8 (frontal channels).",
         12, "blink"),

    Task("MENTAL MATH", "Count backwards from 200 by 7s",
         "200, 193, 186, 179... go as fast as you can.\nMental effort increases beta (13-30 Hz) power.",
         15, "focus"),

    Task("EYES CLOSED RELAX", "Close your eyes and breathe deeply",
         "4 seconds in, 4 seconds out. Mind blank.\nRelaxation increases alpha (8-13 Hz).",
         15, "relax"),

    Task("SILENT READING", "Read the text shown on screen",
         "Text will appear when you press SPACE.\nRead at your normal pace.",
         15, "reading"),

    Task("EYE MOVEMENT", "Follow the dot: left, center, right",
         "A dot will move across the screen.\nFollow it with ONLY your eyes, not your head.",
         12, "movement"),

    Task("DOUBLE CLENCH x3", "Clench jaw twice quickly, pause, repeat 3x",
         "Two quick clenches (tap-tap), 2 second pause.\nRepeat 3 times.",
         12, "clench"),

    Task("VISUALIZATION", "Imagine picking up a glass and drinking",
         "Visualize every detail: weight, cold glass, water.\nKeep eyes on the dot.",
         10, "focus"),
]


def get_band_powers(data_1ch):
    if len(data_1ch) < EEG_SR:
        return {name: 0.0 for name, _, _ in BANDS}
    filtered = data_1ch.copy()
    DataFilter.detrend(filtered, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(filtered, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(filtered, EEG_SR, NoiseTypes.SIXTY.value)
    nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
    psd = DataFilter.get_psd_welch(filtered, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
    return {name: float(DataFilter.get_band_power(psd, lo, hi)) for name, lo, hi in BANDS}


def filter_channel(data):
    if len(data) < 12:
        return data.copy()
    out = data.copy()
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, EEG_SR, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, EEG_SR, NoiseTypes.SIXTY.value)
    return out


def draw_waveform(canvas, data_1ch, x, y, w, h, color, label="", scale=100):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 50), 1)
    if label:
        cv2.putText(canvas, label, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    n = len(data_1ch)
    if n < 2:
        return
    show = min(n, w)
    segment = data_1ch[-show:]
    mid_y = y + h // 2
    cv2.line(canvas, (x, mid_y), (x + w, mid_y), (40, 40, 60), 1)
    pts = []
    for i, val in enumerate(segment):
        px = x + int(i * w / show)
        py = mid_y - int(np.clip(val / scale, -1, 1) * (h // 2 - 4))
        pts.append((px, py))
    if len(pts) > 1:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)


def draw_band_bars(canvas, band_powers, x, y, w, h):
    bar_h = h // len(BANDS) - 2
    for i, (name, _, _) in enumerate(BANDS):
        by = y + i * (bar_h + 2)
        power = band_powers.get(name, 0.0)
        bar_w = int(np.clip(power / 50.0, 0, 1) * w)
        color = BAND_COLORS[name]
        cv2.rectangle(canvas, (x, by), (x + bar_w, by + bar_h), color, -1)
        cv2.rectangle(canvas, (x, by), (x + w, by + bar_h), (40, 40, 60), 1)
        cv2.putText(canvas, f"{name}: {power:.1f}", (x + 4, by + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 200), 1)


def draw_task_visuals(canvas, task, elapsed, sw, sh):
    """Draw task-specific visuals during recording (reading text, dots, etc)."""

    # Countdown + progress bar at top
    remaining = max(0, task.duration - elapsed)
    progress = elapsed / task.duration

    cv2.putText(canvas, task.name, (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
    cv2.putText(canvas, f"{remaining:.0f}s", (sw // 2 - 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

    bar_x, bar_w = 40, sw // 2 - 80
    cv2.rectangle(canvas, (bar_x, 55), (bar_x + bar_w, 61), (40, 40, 60), -1)
    cv2.rectangle(canvas, (bar_x, 55), (bar_x + int(bar_w * progress), 61), (0, 255, 136), -1)

    # REC indicator
    pulse = int(128 + 127 * np.sin(elapsed * 4))
    cv2.circle(canvas, (sw // 2 + 30, 35), 6, (0, 0, pulse), -1)
    cv2.putText(canvas, "REC", (sw // 2 + 42, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    # Instruction text
    cv2.putText(canvas, task.instruction, (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 180), 1)

    mid_x = sw // 4
    mid_y = sh // 2

    if task.category == "baseline":
        # Green fixation dot
        cv2.circle(canvas, (mid_x, mid_y), 6, (0, 255, 100), -1)
        cv2.circle(canvas, (mid_x, mid_y), 12, (0, 255, 100), 1)

    elif task.category == "relax":
        # Breathing guide circle that pulses
        breath_cycle = 8.0  # 4s in + 4s out
        phase = (elapsed % breath_cycle) / breath_cycle
        if phase < 0.5:
            radius = int(20 + 40 * (phase * 2))
            label = "BREATHE IN"
        else:
            radius = int(60 - 40 * ((phase - 0.5) * 2))
            label = "BREATHE OUT"
        cv2.circle(canvas, (mid_x, mid_y), radius, (0, 200, 100), 2)
        cv2.circle(canvas, (mid_x, mid_y), 4, (0, 255, 100), -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(canvas, label, (mid_x - text_size[0] // 2, mid_y + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

    elif task.category == "reading":
        # Show the reading text
        start_y = sh // 2 - len(READING_TEXT) * 20
        for i, line in enumerate(READING_TEXT):
            cv2.putText(canvas, line, (60, start_y + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 240), 1)

    elif task.category == "movement":
        # Dot that moves left → center → right → center, repeating
        cycle = 4.0  # 1s per position
        t = (elapsed % cycle) / cycle
        positions = [0.15, 0.5, 0.85, 0.5]
        idx = int(t * 4) % 4
        frac = (t * 4) % 1.0
        next_idx = (idx + 1) % 4
        # Smooth interpolation between positions
        x_frac = positions[idx] + (positions[next_idx] - positions[idx]) * frac
        dot_x = int(40 + x_frac * (sw // 2 - 80))
        dot_y = mid_y
        cv2.circle(canvas, (dot_x, dot_y), 12, (0, 212, 255), -1)
        cv2.circle(canvas, (dot_x, dot_y), 18, (0, 212, 255), 2)
        # Position markers
        for px in [0.15, 0.5, 0.85]:
            mx = int(40 + px * (sw // 2 - 80))
            cv2.circle(canvas, (mx, dot_y), 3, (60, 60, 80), -1)
        labels = ["LEFT", "CENTER", "RIGHT"]
        pos_names = [0.15, 0.5, 0.85]
        for lbl, px in zip(labels, pos_names):
            mx = int(40 + px * (sw // 2 - 80))
            ts = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(canvas, lbl, (mx - ts[0] // 2, dot_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 120), 1)

    elif task.category == "clench":
        # Show clench timing guide
        if "DOUBLE" in task.name:
            # Double clench: tap-tap every 4 seconds
            cycle_pos = elapsed % 4.0
            if cycle_pos < 0.3:
                label = "CLENCH!"
                color = (0, 100, 255)
            elif cycle_pos < 0.6:
                label = "release"
                color = (100, 100, 140)
            elif cycle_pos < 0.9:
                label = "CLENCH!"
                color = (0, 100, 255)
            else:
                label = "wait..."
                color = (60, 60, 80)
        else:
            # Single clench: every 2 seconds
            cycle_pos = elapsed % 2.0
            if cycle_pos < 1.0:
                label = "CLENCH!"
                color = (0, 100, 255)
            else:
                label = "release"
                color = (100, 100, 140)
        cv2.putText(canvas, label, (mid_x - 80, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    elif task.category == "blink":
        # Blink timing guide
        cycle_pos = elapsed % 2.0
        if cycle_pos < 1.0:
            label = "BLINK"
            color = (0, 200, 255)
        else:
            label = "open"
            color = (100, 100, 140)
        cv2.putText(canvas, label, (mid_x - 60, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    elif task.category == "focus":
        # Fixation dot + subtle reminder
        cv2.circle(canvas, (mid_x, mid_y), 5, (0, 200, 255), -1)
        if "MATH" in task.name:
            cv2.putText(canvas, "200, 193, 186, 179...", (mid_x - 120, mid_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 120), 1)
        elif "VISUAL" in task.name:
            cv2.putText(canvas, "imagine the glass...", (mid_x - 100, mid_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 120), 1)


def show_analysis(recordings, sw, sh):
    cv2.namedWindow("EEG Analysis", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("EEG Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    task_bands = {}
    for rec in recordings:
        avg_bands = {}
        for name, _, _ in BANDS:
            vals = [get_band_powers(rec.eeg_data[ch]).get(name, 0.0) for ch in range(4)]
            avg_bands[name] = np.mean(vals)
        task_bands[rec.task.name] = avg_bands

    baseline_bands = task_bands.get("BASELINE", {name: 1.0 for name, _, _ in BANDS})

    page = 0
    total_pages = 2

    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (20, 6, 6)

        cv2.putText(canvas, "EEG ANALYSIS", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
        cv2.putText(canvas, f"Page {page+1}/{total_pages}  |  LEFT/RIGHT arrows  |  ESC quit",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 104, 148), 1)

        if page == 0:
            cv2.putText(canvas, "BAND POWER BY TASK (vs baseline)", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 208, 232), 1)

            n_tasks = len(recordings)
            row_h = max(30, (sh - 160) // n_tasks)

            for ti, rec in enumerate(recordings):
                ty = 140 + ti * row_h
                cat_colors = {
                    "baseline": (128, 128, 128), "clench": (0, 100, 255),
                    "blink": (255, 200, 0), "focus": (0, 200, 255),
                    "relax": (0, 255, 100), "movement": (255, 100, 200),
                    "reading": (200, 200, 100),
                }
                color = cat_colors.get(rec.task.category, (128, 128, 128))
                cv2.putText(canvas, rec.task.name[:20], (40, ty + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                bar_x = 260
                bar_total_w = sw - 320
                band_w = bar_total_w // len(BANDS)

                for bi, (bname, _, _) in enumerate(BANDS):
                    bx = bar_x + bi * band_w
                    power = task_bands[rec.task.name].get(bname, 0.0)
                    base = baseline_bands.get(bname, 1.0)
                    ratio = power / (base + 0.001)
                    bar_h = int(np.clip(ratio, 0, 3) * (row_h - 8) / 3)
                    bcolor = BAND_COLORS[bname]
                    cv2.rectangle(canvas, (bx + 2, ty + row_h - 4 - bar_h),
                                  (bx + band_w - 2, ty + row_h - 4), bcolor, -1)
                    if ti == 0:
                        cv2.putText(canvas, bname[:5], (bx + 2, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, bcolor, 1)
                    cv2.putText(canvas, f"{ratio:.1f}x", (bx + 2, ty + row_h - 6 - bar_h),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 200), 1)

        elif page == 1:
            cv2.putText(canvas, "RAW WAVEFORMS (filtered 1-50Hz, middle 2s of each task)", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 208, 232), 1)

            n_tasks = len(recordings)
            row_h = max(60, (sh - 160) // n_tasks)
            wf_w = (sw - 280) // 4

            for ci, ch_name in enumerate(CH_NAMES):
                cx = 260 + ci * wf_w
                cv2.putText(canvas, ch_name, (cx + wf_w // 2 - 15, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)

            for ti, rec in enumerate(recordings):
                ty = 140 + ti * row_h
                cat_colors = {
                    "baseline": (128, 128, 128), "clench": (0, 100, 255),
                    "blink": (255, 200, 0), "focus": (0, 200, 255),
                    "relax": (0, 255, 100), "movement": (255, 100, 200),
                    "reading": (200, 200, 100),
                }
                color = cat_colors.get(rec.task.category, (128, 128, 128))
                cv2.putText(canvas, rec.task.name[:18], (10, ty + row_h // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                mid = rec.eeg_data.shape[1] // 2
                show_samples = min(EEG_SR * 2, rec.eeg_data.shape[1])
                start = max(0, mid - show_samples // 2)
                end = start + show_samples

                for ci in range(4):
                    cx = 260 + ci * wf_w
                    segment = rec.eeg_data[ci, start:end]
                    if len(segment) > 10:
                        filtered = filter_channel(segment)
                        ch_color = [(0, 200, 200), (200, 100, 255), (255, 100, 200), (200, 200, 0)][ci]
                        draw_waveform(canvas, filtered, cx, ty, wf_w - 4, row_h - 4,
                                      ch_color, scale=80)

        cv2.imshow("EEG Analysis", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key == 83 or key == 3 or key == ord('d'):  # right arrow
            page = min(page + 1, total_pages - 1)
        elif key == 81 or key == 2 or key == ord('a'):  # left arrow
            page = max(page - 1, 0)

    cv2.destroyAllWindows()


def main():
    from screeninfo import get_monitors
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1470, 956

    print(f"Screen: {sw}x{sh}")

    if not is_bridge_running():
        print("ERROR: Muse bridge not running. Start it first:")
        print("  python3 muse_bridge.py        # real Muse")
        print("  python3 muse_bridge.py --sim   # simulated")
        sys.exit(1)

    eeg = EEGBridgeClient()
    eeg.start()
    print("Connected to Muse bridge")

    # Warm up EEG buffer
    print("Warming up EEG buffer...")
    for _ in range(40):
        eeg.pull()
        time.sleep(0.05)

    cv2.namedWindow("EEG Explorer", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("EEG Explorer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recordings = []
    task_idx = 0

    print(f"\n{len(TASKS)} tasks to complete. Press SPACE to start each one.\n")

    while task_idx < len(TASKS):
        task = TASKS[task_idx]
        phase = "ready"
        rec_start = 0.0
        # Record raw samples, not overlapping windows
        rec_samples = [[] for _ in range(4)]

        while True:
            # Pull new EEG samples
            new_data = eeg.pull()
            window = eeg.get_window(2.0)

            # If recording, collect the NEW samples only (not overlapping windows)
            if phase == "recording" and new_data.shape[1] > 0:
                for ch in range(4):
                    rec_samples[ch].extend(new_data[ch].tolist())

            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            canvas[:] = (20, 6, 6)

            elapsed = time.time() - rec_start if phase == "recording" else 0

            if phase == "ready":
                # Task name
                cv2.putText(canvas, f"TASK {task_idx + 1}/{len(TASKS)}", (40, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 104, 148), 1)
                cv2.putText(canvas, task.name, (sw // 4 - 200, sh // 2 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 255), 2)
                cv2.putText(canvas, "Press SPACE to start", (sw // 4 - 140, sh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                for i, line in enumerate(task.detail.split("\n")):
                    cv2.putText(canvas, line.strip(), (sw // 4 - 250, sh // 2 + 50 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)

            elif phase == "recording":
                # Draw task-specific visuals on the left half
                draw_task_visuals(canvas, task, elapsed, sw, sh)

                # Check if task is done
                if elapsed >= task.duration:
                    phase = "done"
                    eeg_arr = np.array([np.array(ch) for ch in rec_samples])
                    rec = TaskRecording(task=task, start_time=rec_start,
                                        end_time=time.time(), eeg_data=eeg_arr)
                    recordings.append(rec)
                    n_samples = eeg_arr.shape[1] if eeg_arr.ndim == 2 else 0
                    print(f"  [{task.name}] Recorded {n_samples} samples "
                          f"({n_samples/EEG_SR:.1f}s)")
                    task_idx += 1
                    time.sleep(0.3)
                    break

            # Live EEG panel (right 40%)
            panel_x = int(sw * 0.58)
            panel_w = sw - panel_x - 20

            cv2.line(canvas, (panel_x - 10, 0), (panel_x - 10, sh), (30, 30, 50), 1)
            cv2.putText(canvas, "LIVE EEG", (panel_x, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)

            if window.shape[1] >= EEG_SR:
                wf_h = (sh - 120) // 5
                ch_colors = [(0, 200, 200), (200, 100, 255), (255, 100, 200), (200, 200, 0)]
                for ci in range(4):
                    wy = 50 + ci * wf_h
                    filtered = filter_channel(window[ci])
                    draw_waveform(canvas, filtered, panel_x, wy, panel_w, wf_h - 4,
                                  ch_colors[ci], label=CH_NAMES[ci], scale=80)

                avg_bands = {}
                for name, _, _ in BANDS:
                    vals = [get_band_powers(window[ch]).get(name, 0.0) for ch in range(4)]
                    avg_bands[name] = np.mean(vals)
                bar_y = 50 + 4 * wf_h
                draw_band_bars(canvas, avg_bands, panel_x, bar_y, panel_w, sh - bar_y - 30)

            cv2.imshow("EEG Explorer", canvas)
            key = cv2.waitKey(16) & 0xFF

            if key == 27:
                print("Aborted.")
                eeg.stop()
                cv2.destroyAllWindows()
                if recordings:
                    show_analysis(recordings, sw, sh)
                return
            elif key == 32 and phase == "ready":
                phase = "recording"
                rec_start = time.time()
                rec_samples = [[] for _ in range(4)]
                print(f"  [{task.name}] Recording started...")

    cv2.destroyAllWindows()
    eeg.stop()

    print(f"\nAll {len(TASKS)} tasks complete!")
    print("Opening analysis...\n")

    baseline_bands = None
    for rec in recordings:
        avg_bands = {}
        for name, _, _ in BANDS:
            vals = [get_band_powers(rec.eeg_data[ch]).get(name, 0.0) for ch in range(4)]
            avg_bands[name] = np.mean(vals)

        if rec.task.name == "BASELINE":
            baseline_bands = avg_bands

        print(f"  {rec.task.name:20s}  ", end="")
        for name, _, _ in BANDS:
            val = avg_bands[name]
            if baseline_bands:
                ratio = val / (baseline_bands[name] + 0.001)
                print(f"{name}={val:6.1f} ({ratio:.1f}x)  ", end="")
            else:
                print(f"{name}={val:6.1f}  ", end="")
        print()

    show_analysis(recordings, sw, sh)
    print("Done.")


if __name__ == "__main__":
    main()
