#!/usr/bin/env python3
"""Axiom Gaze Box Test — dead simple.

1. Runs eyetrax dense grid calibration (fullscreen OpenCV window)
2. Shows 6 colored boxes
3. Highlights whichever box you're looking at

That's it. No EEG, no WebSocket, no browser. Just gaze → box.
"""

import cv2
import numpy as np
import sys
import time
from eyetrax import GazeEstimator
from eyetrax.calibration import run_dense_grid_calibration
from eyetrax.filters import KalmanEMASmoother, make_kalman
from screeninfo import get_monitors

def get_screen_size():
    try:
        m = get_monitors()[0]
        return m.width, m.height
    except Exception:
        return 1440, 900

def draw_boxes(frame, boxes, gazed_idx, screen_w, screen_h):
    colors = [
        (255, 99, 108),
        (247, 85, 168),
        (255, 212, 0),
        (0, 170, 255),
        (136, 255, 0),
        (157, 107, 255),
    ]
    labels = ["BOX 1", "BOX 2", "BOX 3", "BOX 4", "BOX 5", "BOX 6"]

    for i, (x, y, w, h) in enumerate(boxes):
        color = colors[i % len(colors)]
        is_gazed = (i == gazed_idx)

        overlay = frame.copy()
        if is_gazed:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(labels[i], font, 0.7, 2)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        text_color = (255, 255, 255) if is_gazed else tuple(int(c * 0.7) for c in color)
        cv2.putText(frame, labels[i], (tx, ty), font, 0.7, text_color, 2)

        if is_gazed:
            tag = "LOOKING HERE"
            tag_size = cv2.getTextSize(tag, font, 0.4, 1)[0]
            cv2.putText(frame, tag, (x + (w - tag_size[0]) // 2, y + h - 12),
                        font, 0.4, (255, 255, 255), 1)

def find_camera():
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        ret, _ = cap.read()
        cap.release()
        if ret:
            print(f"Using camera index {idx}")
            return idx
    print("ERROR: No camera found")
    sys.exit(1)

def main():
    screen_w, screen_h = get_screen_size()
    print(f"Screen: {screen_w}x{screen_h}")

    cam_idx = find_camera()

    # tiny_mlp captures the nonlinear eye-to-screen mapping that Ridge can't
    estimator = GazeEstimator(model_name="tiny_mlp")

    print()
    print("=== IMPORTANT ===")
    print("  Keep the calibration window FOCUSED (don't click away).")
    print("  macOS freezes the window if it loses focus.")
    print("  MOVE YOUR HEAD slightly between dots (lean left/right/forward).")
    print("  This teaches the model to handle head movement.")
    print("  Press ESC to abort.")
    print()
    print("Starting 5x5 dense grid calibration (25 points)...")
    print("  Stare at each green dot until the white ring completes.")

    # Dense 5x5 grid = 25 points with serpentine traversal.
    # Faster pulse/capture (0.8s each) so it doesn't take forever.
    run_dense_grid_calibration(
        estimator,
        rows=5,
        cols=5,
        order="serpentine",
        pulse_d=0.8,
        cd_d=0.8,
        camera_index=cam_idx,
    )
    print("Calibration done!")

    estimator.save_model("/Users/anishkataria/axiom/backend/data/gaze_model.pkl")
    print("Model saved to data/gaze_model.pkl")

    # KalmanEMA: Kalman for physics-based prediction + EMA for smoothness
    smoother = KalmanEMASmoother(make_kalman(), ema_alpha=0.3)

    # Box layout (3x2 grid)
    margin_x = int(screen_w * 0.08)
    margin_y = int(screen_h * 0.12)
    gap = 20
    cols, rows = 3, 2
    box_w = (screen_w - 2 * margin_x - (cols - 1) * gap) // cols
    box_h = (screen_h - 2 * margin_y - (rows - 1) * gap - 80) // rows

    boxes = []
    for r in range(rows):
        for c in range(cols):
            bx = margin_x + c * (box_w + gap)
            by = margin_y + 50 + r * (box_h + gap)
            boxes.append((bx, by, box_w, box_h))

    # Main loop
    cap = cv2.VideoCapture(cam_idx)
    cv2.namedWindow("Axiom Gaze Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Axiom Gaze Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    gaze_x, gaze_y = screen_w // 2, screen_h // 2
    gazed_box = -1

    print("Tracking... ESC=quit  R=recalibrate")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink = estimator.extract_features(frame)

        if features is not None and not blink:
            raw = estimator.predict(np.array([features]))[0]
            sx, sy = smoother.step(int(raw[0]), int(raw[1]))
            gaze_x, gaze_y = sx, sy

        # Hit test
        gazed_box = -1
        pad = 30
        for i, (bx, by, bw, bh) in enumerate(boxes):
            if (bx - pad <= gaze_x <= bx + bw + pad and
                    by - pad <= gaze_y <= by + bh + pad):
                gazed_box = i
                break

        # Draw
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (20, 6, 6)

        cv2.putText(canvas, "AXIOM  gaze test", (margin_x, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 208, 232), 2)
        cv2.putText(canvas, f"Gaze: ({gaze_x}, {gaze_y})", (screen_w - 300, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 104, 148), 1)

        draw_boxes(canvas, boxes, gazed_box, screen_w, screen_h)

        cv2.circle(canvas, (gaze_x, gaze_y), 6, (255, 212, 0), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 8, (255, 255, 255), 1)

        footer_text = f"Looking at: BOX {gazed_box + 1}" if gazed_box >= 0 else "Looking at: --"
        cv2.putText(canvas, footer_text, (margin_x, screen_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 104, 148), 1)
        cv2.putText(canvas, "ESC=quit  R=recalibrate", (screen_w - 300, screen_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 84, 120), 1)

        cv2.imshow("Axiom Gaze Test", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            print("Recalibrating...")
            cv2.destroyAllWindows()
            run_dense_grid_calibration(
                estimator, rows=5, cols=5, order="serpentine",
                pulse_d=0.8, cd_d=0.8, camera_index=cam_idx,
            )
            estimator.save_model("/Users/anishkataria/axiom/backend/data/gaze_model.pkl")
            cv2.namedWindow("Axiom Gaze Test", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Axiom Gaze Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    estimator.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
