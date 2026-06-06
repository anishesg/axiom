"""Overlay controller — spawns and communicates with the overlay process."""

import subprocess
import json
import os
import sys


class OverlayController:
    def __init__(self):
        self._proc = None
        self._ready = False

    def start(self):
        script = os.path.join(os.path.dirname(__file__), "overlay.py")
        python = os.path.join(os.path.dirname(sys.executable), "python3")
        if not os.path.exists(python):
            python = sys.executable

        self._proc = subprocess.Popen(
            [python, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        line = self._proc.stdout.readline().strip()
        if line == "OVERLAY_READY":
            self._ready = True
            print("  [OVERLAY] Ready", flush=True)
        else:
            print(f"  [OVERLAY] Unexpected: {line}", flush=True)

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            self._proc = None
            self._ready = False

    def _send(self, cmd: dict):
        if not self._ready or not self._proc or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps(cmd) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._ready = False

    def presence(self, active: bool, engagement: float = 0.0):
        self._send({"cmd": "presence", "active": active, "engagement": engagement})

    def intent_ring(self, active: bool, progress: float = 0.0,
                    x: float = 0, y: float = 0):
        self._send({"cmd": "intent_ring", "active": active,
                     "progress": progress, "x": x, "y": y})

    def gaze(self, x: float, y: float):
        self._send({"cmd": "gaze", "x": x, "y": y})

    def flash(self, color: str = "green"):
        self._send({"cmd": "flash", "color": color})

    def mode(self, mode: str):
        self._send({"cmd": "mode", "mode": mode})

    def status(self, text: str):
        self._send({"cmd": "status", "text": text})

    def calibration_target(self, x: float = None, y: float = None):
        if x is not None and y is not None:
            self._send({"cmd": "calibration_target", "x": x, "y": y})
        else:
            self._send({"cmd": "calibration_target"})

    @property
    def running(self) -> bool:
        return self._ready and self._proc and self._proc.poll() is None
