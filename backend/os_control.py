"""Axiom OS Control — macOS screen reading, actions, and overlay.

Uses pyobjc for native macOS APIs:
  - NSWorkspace: active app, running apps
  - Quartz/CoreGraphics: window list, screen capture, synthetic events
  - AXUIElement: UI element tree (buttons, text fields, etc.)
  - PyAutoGUI: click, scroll, type (backed by Quartz)

Permissions required:
  - Accessibility (System Settings > Privacy > Accessibility)
  - Screen Recording (System Settings > Privacy > Screen Recording)
"""

import subprocess
from dataclasses import dataclass, field
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

try:
    from AppKit import NSWorkspace, NSRunningApplication
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    HAS_PYOBJC = True
except ImportError:
    HAS_PYOBJC = False


@dataclass
class WindowInfo:
    name: str = ""
    owner: str = ""
    pid: int = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    layer: int = 0
    is_on_screen: bool = True


@dataclass
class ScreenState:
    active_app: str = ""
    active_app_pid: int = 0
    focused_window: str = ""
    windows: list = field(default_factory=list)
    dock_apps: list = field(default_factory=list)


class OSControl:
    """Read screen state and perform OS actions."""

    def get_screen_state(self) -> ScreenState:
        state = ScreenState()
        if not HAS_PYOBJC:
            return state

        ws = NSWorkspace.sharedWorkspace()
        active = ws.activeApplication()
        if active:
            state.active_app = active.get("NSApplicationName", "")
            state.active_app_pid = active.get("NSApplicationProcessIdentifier", 0)

        windows = CGWindowListCopyWindowInfo(
            kCGWindowListExcludeDesktopElements | kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
        if windows:
            for w in windows:
                bounds = w.get("kCGWindowBounds", {})
                info = WindowInfo(
                    name=w.get("kCGWindowName", "") or "",
                    owner=w.get("kCGWindowOwnerName", "") or "",
                    pid=w.get("kCGWindowOwnerPID", 0),
                    x=int(bounds.get("X", 0)),
                    y=int(bounds.get("Y", 0)),
                    width=int(bounds.get("Width", 0)),
                    height=int(bounds.get("Height", 0)),
                    layer=w.get("kCGWindowLayer", 0),
                )
                if info.width > 50 and info.height > 50 and info.layer == 0:
                    state.windows.append(info)

            if state.windows:
                state.focused_window = state.windows[0].name

        state.dock_apps = self._get_dock_apps()
        return state

    def _get_dock_apps(self) -> list[str]:
        """Get apps currently in the Dock."""
        if not HAS_PYOBJC:
            return []
        ws = NSWorkspace.sharedWorkspace()
        running = ws.runningApplications()
        return [
            app.localizedName()
            for app in running
            if app.activationPolicy() == 0  # NSApplicationActivationPolicyRegular
            and app.localizedName()
        ]

    def get_cursor_position(self) -> tuple[int, int]:
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def click(self, x: int, y: int):
        pyautogui.click(x, y)

    def double_click(self, x: int, y: int):
        pyautogui.doubleClick(x, y)

    def scroll(self, amount: int, x: int = None, y: int = None):
        pyautogui.scroll(amount, x=x, y=y)

    def type_text(self, text: str, interval: float = 0.02):
        pyautogui.write(text, interval=interval)

    def hotkey(self, *keys):
        pyautogui.hotkey(*keys)

    def switch_app(self, app_name: str):
        """Activate an app by name using AppleScript (most reliable on macOS)."""
        subprocess.run([
            "osascript", "-e",
            f'tell application "{app_name}" to activate',
        ], capture_output=True, timeout=5)

    def open_app(self, app_name: str):
        subprocess.run(["open", "-a", app_name], capture_output=True, timeout=5)

    def close_window(self):
        pyautogui.hotkey("command", "w")

    def undo(self):
        pyautogui.hotkey("command", "z")

    def cmd_tab(self):
        pyautogui.hotkey("command", "tab")

    def get_element_at(self, x: int, y: int) -> dict:
        """Get UI element at screen coordinates using accessibility API."""
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCopyElementAtPosition,
                AXUIElementCopyAttributeValue,
            )
            system = AXUIElementCreateSystemWide()
            err, element = AXUIElementCopyElementAtPosition(system, float(x), float(y))
            if err != 0 or element is None:
                return {"error": "no element", "code": err}

            result = {}
            for attr in ["AXRole", "AXTitle", "AXValue", "AXDescription",
                         "AXRoleDescription"]:
                err, val = AXUIElementCopyAttributeValue(element, attr)
                if err == 0 and val is not None:
                    result[attr] = str(val)
            return result
        except Exception as e:
            return {"error": str(e)}

    def map_gaze_to_context(self, gaze_x: float, gaze_y: float,
                            screen_state: ScreenState) -> dict:
        """Map gaze position to a meaningful screen context."""
        context = {
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "active_app": screen_state.active_app,
            "region": self._classify_gaze_region(gaze_x, gaze_y),
        }

        for w in screen_state.windows:
            if (w.x <= gaze_x <= w.x + w.width and
                    w.y <= gaze_y <= w.y + w.height):
                context["window"] = w.name
                context["window_app"] = w.owner
                break

        element = self.get_element_at(int(gaze_x), int(gaze_y))
        if "error" not in element:
            context["element_role"] = element.get("AXRole", "")
            context["element_title"] = element.get("AXTitle", "")
            context["element_value"] = element.get("AXValue", "")[:100] if element.get("AXValue") else ""

        return context

    def _classify_gaze_region(self, x: float, y: float) -> str:
        screen_h = 900  # logical points on typical iMac
        if y < 25:
            return "menu_bar"
        if y > screen_h - 70:
            return "dock"
        col = "left" if x < 853 else ("right" if x > 1707 else "center")
        row = "top" if y < 300 else ("bottom" if y > 600 else "middle")
        return f"{row}-{col}"
