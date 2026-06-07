#!/usr/bin/env python3
"""Gaze-controlled servo — the ultrasonic sensor looks where you look.

Pipeline:
  1. Eyetrax webcam gaze calibration (5x5 grid)
  2. Continuous gaze tracking at ~30Hz
  3. Map gaze X position → servo angle (0-180)
  4. Stream servo commands + radar readings over serial

Usage:
  python3 gaze_servo.py [--port /dev/cu.usbmodem112301] [--camera 0]
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import serial

from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from screeninfo import get_monitors


def get_screen():
    try:
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return 1470, 956


def find_camera(preferred=None):
    candidates = [preferred] if preferred is not None else [1, 0]
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        ret, _ = cap.read()
        cap.release()
        if ret:
            return idx
    print("ERROR: No camera found")
    sys.exit(1)


def connect_arduino(port, baud=9600):
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2)  # Arduino resets on serial connect
    # Drain startup messages
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  Arduino: {line}")
    return ser


def send_servo(ser, angle):
    angle = max(0, min(180, int(angle)))
    ser.write(f"S,{angle}\n".encode())


def resume_sweep(ser):
    ser.write(b"R\n")


def read_serial(ser):
    """Non-blocking read of any available serial lines."""
    lines = []
    while ser.in_waiting:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
        except Exception:
            break
    return lines


def main():
    parser = argparse.ArgumentParser(description="Gaze-controlled servo")
    parser.add_argument("--port", default="/dev/cu.usbmodem112301")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--skip-cal", action="store_true", help="Skip calibration (for testing)")
    args = parser.parse_args()

    sw, sh = get_screen()
    print(f"Screen: {sw}x{sh}")

    # Connect Arduino
    print(f"\nConnecting to Arduino on {args.port}...")
    ser = connect_arduino(args.port)
    ser.write(b"P\n")
    time.sleep(0.5)
    for line in read_serial(ser):
        print(f"  Arduino: {line}")
    print("Arduino connected.\n")

    # Find camera
    cam_idx = find_camera(args.camera)
    print(f"Using camera {cam_idx}")

    # Init gaze estimator
    gaze = GazeEstimator(model_name="tiny_mlp")

    if not args.skip_cal:
        print("\n=== GAZE CALIBRATION ===")
        print("  Follow the dots with your eyes. Move head slightly between dots.")
        run_dense_grid_calibration(
            gaze, rows=5, cols=5, order="serpentine",
            pulse_d=0.8, cd_d=0.8, camera_index=cam_idx,
        )
        print("  Calibration done.\n")

    smoother = KalmanEMASmoother(make_kalman(), ema_alpha=0.3)

    # Servo smoothing — avoid jitter
    servo_history = deque(maxlen=5)
    current_servo_angle = 90
    last_sent_angle = -1

    # Radar data storage
    radar_map = {}

    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Axiom Gaze Servo", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Axiom Gaze Servo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    gaze_x, gaze_y = sw // 2, sh // 2

    print("=== LIVE — Look around to control the servo. ESC to quit. ===\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Gaze tracking
        features, blink = gaze.extract_features(frame)
        if features is not None and not blink:
            raw = gaze.predict(np.array([features]))[0]
            sx, sy = smoother.step(int(raw[0]), int(raw[1]))
            gaze_x = max(0, min(sw, sx))
            gaze_y = max(0, min(sh, sy))

        # Map gaze X → servo angle
        # Screen left (x=0) → servo 180 (looking left from car's perspective)
        # Screen right (x=sw) → servo 0
        # This mirrors: when you look left on screen, sensor points left
        target_angle = 180.0 - (gaze_x / sw) * 180.0
        servo_history.append(target_angle)
        smoothed_angle = int(np.mean(servo_history))

        # Only send if angle changed enough (reduces serial spam)
        if abs(smoothed_angle - last_sent_angle) >= 2:
            send_servo(ser, smoothed_angle)
            last_sent_angle = smoothed_angle
            current_servo_angle = smoothed_angle

        # Read radar data from Arduino
        for line in read_serial(ser):
            if line.startswith("R,"):
                parts = line.split(",")
                if len(parts) == 3:
                    try:
                        angle = int(parts[1])
                        dist = float(parts[2])
                        if dist > 0.5:
                            radar_map[angle] = dist
                    except ValueError:
                        pass

        # --- Draw ---
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[:] = (15, 5, 5)

        # Header
        cv2.putText(canvas, "AXIOM GAZE SERVO", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 208, 232), 2)
        cv2.putText(canvas, f"Servo: {current_servo_angle}deg", (sw - 200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)

        # Radar display (bottom half, centered)
        radar_cx = sw // 2
        radar_cy = sh - 60
        radar_r = min(sw // 3, sh // 3)

        # Draw radar arcs
        for r_frac in [0.25, 0.5, 0.75, 1.0]:
            r = int(radar_r * r_frac)
            cv2.ellipse(canvas, (radar_cx, radar_cy), (r, r), 0, 180, 360, (25, 25, 40), 1)

        # Draw radar rays every 30 degrees
        for deg in range(0, 181, 30):
            rad = np.radians(180 - deg)
            ex = int(radar_cx + radar_r * np.cos(rad))
            ey = int(radar_cy - radar_r * np.sin(rad))
            cv2.line(canvas, (radar_cx, radar_cy), (ex, ey), (25, 25, 40), 1)
            cv2.putText(canvas, f"{deg}", (ex - 10, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 90), 1)

        # Plot radar points
        max_range = 200.0
        for angle, dist in radar_map.items():
            if dist < 0.5:
                continue
            r_px = int((dist / max_range) * radar_r)
            r_px = min(r_px, radar_r)
            rad = np.radians(180 - angle)
            px = int(radar_cx + r_px * np.cos(rad))
            py = int(radar_cy - r_px * np.sin(rad))

            # Color by distance: green=far, yellow=mid, red=close
            if dist < 15:
                color = (0, 0, 255)
            elif dist < 40:
                color = (0, 180, 255)
            elif dist < 80:
                color = (0, 255, 200)
            else:
                color = (0, 255, 80)

            cv2.circle(canvas, (px, py), 3, color, -1)

        # Draw servo direction line (where it's currently pointing)
        servo_rad = np.radians(180 - current_servo_angle)
        sx_line = int(radar_cx + radar_r * np.cos(servo_rad))
        sy_line = int(radar_cy - radar_r * np.sin(servo_rad))
        cv2.line(canvas, (radar_cx, radar_cy), (sx_line, sy_line), (0, 212, 255), 2)

        # Distance readout at current servo angle
        current_dist = radar_map.get(current_servo_angle, -1)
        if current_dist > 0:
            dist_text = f"{current_dist:.0f}cm"
        else:
            # Try nearest angle
            nearest = min(radar_map.keys(), key=lambda a: abs(a - current_servo_angle), default=None)
            if nearest is not None and abs(nearest - current_servo_angle) < 5:
                dist_text = f"~{radar_map[nearest]:.0f}cm"
            else:
                dist_text = "--"
        cv2.putText(canvas, dist_text, (radar_cx - 25, radar_cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)

        # Gaze dot (top area of screen)
        cv2.circle(canvas, (gaze_x, gaze_y), 8, (0, 212, 255), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 10, (255, 255, 255), 1)

        # Gaze X indicator bar (shows mapping to servo)
        bar_y = sh // 2 - 30
        bar_x = 40
        bar_w = sw - 80
        bar_h = 12
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 50), -1)
        gaze_norm = gaze_x / sw
        marker_x = bar_x + int(bar_w * gaze_norm)
        cv2.rectangle(canvas, (marker_x - 3, bar_y - 2), (marker_x + 3, bar_y + bar_h + 2),
                      (0, 212, 255), -1)
        cv2.putText(canvas, "0", (bar_x + bar_w + 5, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 120), 1)
        cv2.putText(canvas, "180", (bar_x - 30, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 120), 1)
        cv2.putText(canvas, "GAZE -> SERVO", (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 140), 1)

        cv2.imshow("Axiom Gaze Servo", canvas)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("r"):
            # Recalibrate
            cap.release()
            cv2.destroyAllWindows()
            run_dense_grid_calibration(
                gaze, rows=5, cols=5, order="serpentine",
                pulse_d=0.8, cd_d=0.8, camera_index=cam_idx,
            )
            smoother = KalmanEMASmoother(make_kalman(), ema_alpha=0.3)
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Axiom Gaze Servo", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Axiom Gaze Servo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord("s"):
            # Resume auto-sweep mode
            resume_sweep(ser)
            print("  Resumed auto-sweep")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    resume_sweep(ser)
    ser.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
