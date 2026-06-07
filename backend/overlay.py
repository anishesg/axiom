#!/usr/bin/env python3
"""Axiom Overlay — transparent always-on-top macOS window.

Runs as a standalone process. Receives JSON commands on stdin.
NSApplication.run() on main thread for proper macOS rendering.

Commands (JSON lines on stdin):
  {"cmd": "presence", "active": true, "engagement": 0.8}
  {"cmd": "intent_ring", "active": true, "progress": 0.5, "x": 400, "y": 300}
  {"cmd": "gaze", "x": 500, "y": 400}
  {"cmd": "flash", "color": "green"}
  {"cmd": "mode", "mode": "tracking"}
  {"cmd": "status", "text": "CALIBRATING..."}
  {"cmd": "calibration_target", "x": 640, "y": 360}
  {"cmd": "calibration_target"}  (clear)
"""

import sys
import json
import threading
import math

import AppKit
import objc
from Cocoa import NSColor, NSFont, NSMakeRect, NSMakePoint
from Quartz import CGDisplayBounds, CGMainDisplayID


class State:
    def __init__(self):
        self.presence_active = False
        self.presence_engagement = 0.0
        self.intent_ring_active = False
        self.intent_ring_progress = 0.0
        self.intent_ring_x = 0.0
        self.intent_ring_y = 0.0
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.gaze_active = False
        self.gaze_trail: list[tuple[float, float, float]] = []
        self.action_flash = ""
        self.action_flash_alpha = 0.0
        self.mode = "passive"
        self.calibration_target: tuple[float, float] | None = None
        self.status_text = ""
        self.lock = threading.Lock()

    def handle(self, cmd: dict):
        with self.lock:
            c = cmd.get("cmd", "")
            if c == "presence":
                self.presence_active = cmd.get("active", False)
                self.presence_engagement = cmd.get("engagement", 0.0)
            elif c == "intent_ring":
                self.intent_ring_active = cmd.get("active", False)
                self.intent_ring_progress = cmd.get("progress", 0.0)
                self.intent_ring_x = cmd.get("x", 0.0)
                self.intent_ring_y = cmd.get("y", 0.0)
            elif c == "gaze":
                self.gaze_x = cmd.get("x", 0.0)
                self.gaze_y = cmd.get("y", 0.0)
                self.gaze_active = True
                self.gaze_trail.append((self.gaze_x, self.gaze_y, 1.0))
                if len(self.gaze_trail) > 30:
                    self.gaze_trail.pop(0)
                self.gaze_trail = [
                    (gx, gy, a * 0.92)
                    for gx, gy, a in self.gaze_trail if a > 0.05
                ]
            elif c == "flash":
                self.action_flash = cmd.get("color", "green")
                self.action_flash_alpha = 1.0
            elif c == "mode":
                self.mode = cmd.get("mode", "passive")
            elif c == "status":
                self.status_text = cmd.get("text", "")
            elif c == "calibration_target":
                if "x" in cmd and "y" in cmd:
                    self.calibration_target = (cmd["x"], cmd["y"])
                else:
                    self.calibration_target = None


S = State()


class OverlayView(AppKit.NSView):
    _tick = 0

    def isFlipped(self):
        return True

    def drawRect_(self, rect):
        self._tick += 1
        t = self._tick / 60.0

        with S.lock:
            self._draw_presence(t)
            self._draw_gaze_trail()
            self._draw_intent_ring(t)
            self._draw_action_flash()
            self._draw_status_text()
            self._draw_calibration_target(t)

    def _draw_presence(self, t):
        if not S.presence_active:
            return
        screen = self.frame()
        x = screen.size.width - 50
        y = 50
        eng = max(0.3, S.presence_engagement)
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        radius = 8 + pulse * 4 * eng

        NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, 0.15 * eng).set()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(x - radius - 6, y - radius - 6,
                       (radius + 6) * 2, (radius + 6) * 2)
        ).fill()

        NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, 0.5 + 0.5 * eng).set()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(x - radius, y - radius, radius * 2, radius * 2)
        ).fill()

        mode_colors = {
            "passive": (0.5, 0.5, 0.6),
            "tracking": (0.0, 0.9, 0.54),
            "calibrating": (1.0, 0.75, 0.15),
            "sleeping": (0.3, 0.3, 0.45),
        }
        r, g, b = mode_colors.get(S.mode, (0.5, 0.5, 0.6))
        attrs = {
            AppKit.NSFontAttributeName: NSFont.boldSystemFontOfSize_(9),
            AppKit.NSForegroundColorAttributeName:
                NSColor.colorWithRed_green_blue_alpha_(r, g, b, 0.7),
        }
        label = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            S.mode.upper(), attrs
        )
        label.drawAtPoint_(NSMakePoint(x - 22, y + 18))

    def _draw_gaze_trail(self):
        if not S.gaze_active:
            return
        for gx, gy, alpha in S.gaze_trail:
            if alpha < 0.05:
                continue
            r = 3 + alpha * 3
            NSColor.colorWithRed_green_blue_alpha_(0.3, 0.56, 0.97, alpha * 0.35).set()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(gx - r, gy - r, r * 2, r * 2)
            ).fill()
        cx, cy = S.gaze_x, S.gaze_y
        NSColor.colorWithRed_green_blue_alpha_(0.3, 0.56, 0.97, 0.5).set()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - 6, cy - 6, 12, 12)
        ).fill()

    def _draw_intent_ring(self, t):
        if not S.intent_ring_active:
            return
        cx, cy = S.intent_ring_x, S.intent_ring_y
        progress = S.intent_ring_progress
        radius = 26

        NSColor.colorWithRed_green_blue_alpha_(0.3, 0.3, 0.4, 0.25).set()
        ring = AppKit.NSBezierPath.bezierPath()
        ring.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
            NSMakePoint(cx, cy), radius, 0, 360, False
        )
        ring.setLineWidth_(3.0)
        ring.stroke()

        if progress > 0:
            green = min(1.0, progress)
            NSColor.colorWithRed_green_blue_alpha_(1.0 - green, green, 0.2, 0.85).set()
            arc = AppKit.NSBezierPath.bezierPath()
            arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                NSMakePoint(cx, cy), radius, 90, 90 - progress * 360, True
            )
            arc.setLineWidth_(3.5)
            arc.stroke()

        if progress >= 1.0:
            pulse = 0.5 + 0.5 * math.sin(t * 8)
            NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, 0.25 * pulse).set()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - radius - 6, cy - radius - 6,
                           (radius + 6) * 2, (radius + 6) * 2)
            ).fill()

    def _draw_action_flash(self):
        if S.action_flash_alpha <= 0:
            return
        a = S.action_flash_alpha
        if S.action_flash == "green":
            NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, a * 0.06).set()
        elif S.action_flash == "red":
            NSColor.colorWithRed_green_blue_alpha_(0.97, 0.3, 0.42, a * 0.06).set()
        else:
            return
        AppKit.NSBezierPath.fillRect_(self.frame())
        S.action_flash_alpha = max(0, a - 0.025)

    def _draw_status_text(self):
        if not S.status_text:
            return
        screen = self.frame()
        attrs = {
            AppKit.NSFontAttributeName: NSFont.boldSystemFontOfSize_(13),
            AppKit.NSForegroundColorAttributeName:
                NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.65),
        }
        text = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            S.status_text, attrs
        )
        size = text.size()
        x = (screen.size.width - size.width) / 2
        text.drawAtPoint_(NSMakePoint(x, screen.size.height - 55))

    def _draw_calibration_target(self, t):
        if S.calibration_target is None:
            return
        cx, cy = S.calibration_target
        pulse = 0.5 + 0.5 * math.sin(t * 4)

        r_outer = 22 + pulse * 6
        NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, 0.2 + pulse * 0.1).set()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - r_outer, cy - r_outer, r_outer * 2, r_outer * 2)
        ).fill()

        NSColor.colorWithRed_green_blue_alpha_(0.0, 0.9, 0.54, 0.9).set()
        AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - 5, cy - 5, 10, 10)
        ).fill()

        NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.35).set()
        line = AppKit.NSBezierPath.bezierPath()
        line.moveToPoint_(NSMakePoint(cx - 14, cy))
        line.lineToPoint_(NSMakePoint(cx + 14, cy))
        line.moveToPoint_(NSMakePoint(cx, cy - 14))
        line.lineToPoint_(NSMakePoint(cx, cy + 14))
        line.setLineWidth_(1.0)
        line.stroke()


def stdin_reader():
    """Read JSON commands from stdin in a background thread."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            S.handle(cmd)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"overlay error: {e}", file=sys.stderr)


def main():
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    display = CGDisplayBounds(CGMainDisplayID())
    frame = NSMakeRect(
        display.origin.x, display.origin.y,
        display.size.width, display.size.height,
    )

    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        AppKit.NSWindowStyleMaskBorderless,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    win.setLevel_(AppKit.NSScreenSaverWindowLevel)
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setIgnoresMouseEvents_(True)
    win.setHasShadow_(False)
    win.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorIgnoresCycle
    )

    view = OverlayView.alloc().initWithFrame_(frame)
    win.setContentView_(view)
    win.makeKeyAndOrderFront_(None)

    # 60fps refresh timer
    def refresh(_timer):
        view.setNeedsDisplay_(True)

    timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        1.0 / 60.0, True, refresh
    )
    AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(
        timer, AppKit.NSDefaultRunLoopMode
    )

    # Read commands from stdin in background
    reader = threading.Thread(target=stdin_reader, daemon=True)
    reader.start()

    print("OVERLAY_READY", flush=True)
    app.run()


if __name__ == "__main__":
    main()
