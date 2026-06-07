"""Gmail Controller — Playwright-based Gmail automation for BCI gaze control.

Uses a persistent Chromium browser context so the user stays logged in.
The gaze tracker provides screen coordinates; this module converts them to
viewport coords and maps them to interactive Gmail elements scanned via JS.

Browser window: positioned at (0, 0), viewport 1400x900.
Chrome toolbar height on macOS: ~88px (used for screen→viewport conversion).
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GMAIL_URL = "https://mail.google.com/"
USER_DATA_DIR = "/Users/anish/muse-brain/data/chrome_profile"
WINDOW_X = 0
WINDOW_Y = 0
VIEWPORT_WIDTH = 1400
VIEWPORT_HEIGHT = 900
CHROME_TOOLBAR_HEIGHT = 88  # macOS Chrome toolbar (address bar + tabs)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GmailElement:
    id: str
    type: str          # email_row, star_button, archive_button, delete_button,
                       # reply_button, compose_button, search_input,
                       # back_button, nav_item
    label: str
    bbox: dict         # {x, y, width, height} in VIEWPORT coordinates
    metadata: dict     # sender, subject, unread, href, etc.


@dataclass
class GmailContext:
    view: str                      # "inbox", "email", "compose", "search"
    elements: list[GmailElement]   # all interactive elements on screen
    selected_email: dict           # non-empty when viewing an email
    unread_count: int
    window_bounds: dict            # {x, y, width, height} of browser on screen


# ──────────────────────────────────────────────────────────────────────────────
# JavaScript injected into the page to collect interactive elements
# ──────────────────────────────────────────────────────────────────────────────

_SCAN_JS = """
(() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const sel = [
        'a', 'button',
        '[role="button"]', '[role="row"]', '[role="link"]',
        '[role="checkbox"]', '[role="listitem"]',
        '[tabindex]', 'input', 'textarea',
    ].join(',');

    const nodes = Array.from(document.querySelectorAll(sel));
    const results = [];

    for (const el of nodes) {
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        if (r.right < 0 || r.bottom < 0) continue;
        if (r.left > vw || r.top > vh) continue;

        // Skip tiny 1×1 spacers and invisible elements
        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none') continue;
        if (style.opacity === '0') continue;

        const rawText = (el.innerText || el.textContent || '').trim();
        results.push({
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            ariaLabel: el.getAttribute('aria-label') || '',
            title: el.getAttribute('title') || '',
            text: rawText.slice(0, 200),
            href: el.getAttribute('href') || '',
            className: el.className || '',
            inputType: el.getAttribute('type') || '',
            x: r.left,
            y: r.top,
            width: r.width,
            height: r.height,
        });
    }
    return JSON.stringify(results);
})()
"""


# ──────────────────────────────────────────────────────────────────────────────
# Element classifier
# ──────────────────────────────────────────────────────────────────────────────

def _classify_element(el: dict) -> Optional[str]:
    """Return an element type string or None if the element should be ignored."""
    aria = el.get("ariaLabel", "").lower()
    title = el.get("title", "").lower()
    text = el.get("text", "").lower()
    role = el.get("role", "").lower()
    class_name = el.get("className", "")
    tag = el.get("tag", "")
    href = el.get("href", "")

    # Search input
    if tag in ("input", "textarea") and (
        "search" in aria or "search" in title
    ):
        return "search_input"

    # Compose button
    if "compose" in aria or "compose" in text:
        return "compose_button"

    # Archive button
    if "archive" in aria or "archive" in title:
        return "archive_button"

    # Delete / Trash
    if "delete" in aria or "delete" in title or "trash" in aria:
        return "delete_button"

    # Star / unstar
    if "star" in aria or "starred" in aria or "not starred" in aria or "star" in title:
        return "star_button"

    # Reply
    if "reply" in aria or "reply" in title:
        return "reply_button"

    # Back (to inbox or prev/next)
    if "back" in aria or "back to inbox" in aria or "newer" in aria or "older" in aria:
        return "back_button"

    # Navigation items (sidebar links)
    nav_keywords = ("inbox", "starred", "snoozed", "sent", "drafts", "spam",
                    "trash", "all mail", "important")
    if any(kw in aria for kw in nav_keywords) or any(kw in text for kw in nav_keywords):
        if role in ("link", "listitem", "menuitem", "") and tag in ("a", "li", "div", "span"):
            return "nav_item"

    # Email rows — Gmail uses class 'zA' for rows, 'zE' for unread rows
    if "zA" in class_name or "zE" in class_name:
        return "email_row"
    if role == "row":
        return "email_row"

    return None


def _make_element(raw: dict) -> Optional[GmailElement]:
    """Convert a raw JS element dict to a GmailElement, or None if unclassified."""
    etype = _classify_element(raw)
    if etype is None:
        return None

    label = (
        raw.get("ariaLabel")
        or raw.get("title")
        or raw.get("text", "")[:80]
        or raw.get("href", "")[:80]
    )

    metadata: dict = {}
    if etype == "email_row":
        # Try to extract sender and subject from the text blob
        text = raw.get("text", "")
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if parts:
            metadata["sender"] = parts[0]
        if len(parts) > 1:
            metadata["subject"] = parts[1]
        if len(parts) > 2:
            metadata["snippet"] = " ".join(parts[2:])[:120]
        # Unread: Gmail marks unread rows with class 'zE' (bold row)
        metadata["unread"] = "zE" in raw.get("className", "")
    elif etype == "nav_item":
        metadata["href"] = raw.get("href", "")

    return GmailElement(
        id=str(uuid.uuid4()),
        type=etype,
        label=label,
        bbox={
            "x": raw["x"],
            "y": raw["y"],
            "width": raw["width"],
            "height": raw["height"],
        },
        metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class GmailController:
    """Playwright-based Gmail controller for gaze/BCI interaction."""

    def __init__(
        self,
        user_data_dir: str = USER_DATA_DIR,
        window_x: int = WINDOW_X,
        window_y: int = WINDOW_Y,
        viewport_width: int = VIEWPORT_WIDTH,
        viewport_height: int = VIEWPORT_HEIGHT,
        toolbar_height: int = CHROME_TOOLBAR_HEIGHT,
    ):
        self._user_data_dir = user_data_dir
        self._window_x = window_x
        self._window_y = window_y
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._toolbar_height = toolbar_height

        self._playwright = None
        self._browser: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._elements: list[GmailElement] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        """Launch the persistent Chromium browser and navigate to Gmail."""
        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=self._user_data_dir,
            headless=False,
            args=[
                f"--window-position={self._window_x},{self._window_y}",
                f"--window-size={self._viewport_width},{self._viewport_height + self._toolbar_height}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions-except=",
            ],
            viewport={"width": self._viewport_width, "height": self._viewport_height},
            no_viewport=False,
        )

        pages = self._browser.pages
        self._page = pages[0] if pages else await self._browser.new_page()
        await self._page.goto(GMAIL_URL, wait_until="domcontentloaded")
        await self._page.wait_for_load_state("networkidle", timeout=15000)

    async def stop(self):
        """Close browser and clean up Playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._page = None
        self._playwright = None

    # ── Element scanning ───────────────────────────────────────────────────────

    async def scan_elements(self) -> list[GmailElement]:
        """Scan the page for all interactive elements and their bounding boxes."""
        if self._page is None:
            return []

        try:
            raw_json: str = await self._page.evaluate(_SCAN_JS)
            raw_list: list[dict] = json.loads(raw_json)
        except Exception:
            return []

        elements: list[GmailElement] = []
        for raw in raw_list:
            el = _make_element(raw)
            if el is not None:
                elements.append(el)

        self._elements = elements
        return elements

    # ── Context ────────────────────────────────────────────────────────────────

    async def get_context(self) -> GmailContext:
        """Return full Gmail state: current view, elements, selected email, etc."""
        elements = await self.scan_elements()
        view = await self._detect_view()
        selected = {}
        if view == "email":
            selected = await self._get_selected_email_info()
        unread = await self._count_unread()
        window_bounds = {
            "x": self._window_x,
            "y": self._window_y,
            "width": self._viewport_width,
            "height": self._viewport_height + self._toolbar_height,
        }
        return GmailContext(
            view=view,
            elements=elements,
            selected_email=selected,
            unread_count=unread,
            window_bounds=window_bounds,
        )

    async def _detect_view(self) -> str:
        """Heuristically determine whether we're in inbox, email, compose, or search."""
        if self._page is None:
            return "unknown"
        url = self._page.url
        if "#inbox" in url or url.rstrip("/").endswith("mail.google.com"):
            pass  # fall through to DOM check
        try:
            view = await self._page.evaluate("""
                () => {
                    // Compose dialog open
                    if (document.querySelector('[role="dialog"]')) return 'compose';
                    // Viewing an individual email thread
                    if (document.querySelector('[data-legacy-thread-id]')) return 'email';
                    if (document.querySelector('.aeF')) return 'email';
                    // Search results page
                    if (document.querySelector('[aria-label="Search results"]')) return 'search';
                    // Search box has a value
                    const sb = document.querySelector('input[aria-label="Search mail"]');
                    if (sb && sb.value.trim()) return 'search';
                    return 'inbox';
                }
            """)
            return view
        except Exception:
            return "inbox"

    async def _get_selected_email_info(self) -> dict:
        """Extract subject, sender, and body preview of the open email thread."""
        if self._page is None:
            return {}
        try:
            info = await self._page.evaluate("""
                () => {
                    const subject = document.querySelector('h2.hP, [data-legacy-thread-id] h2, .nH .ha h2');
                    const sender = document.querySelector('.gD');
                    const body = document.querySelector('.a3s.aiL, .ii.gt div');
                    return {
                        subject: subject ? subject.innerText.trim() : '',
                        sender: sender ? (sender.getAttribute('email') || sender.innerText.trim()) : '',
                        body_preview: body ? body.innerText.trim().slice(0, 300) : '',
                    };
                }
            """)
            return info
        except Exception:
            return {}

    async def _count_unread(self) -> int:
        """Return number of unread emails visible in the inbox list."""
        return sum(1 for el in self._elements
                   if el.type == "email_row" and el.metadata.get("unread"))

    # ── Coordinate mapping ─────────────────────────────────────────────────────

    def screen_to_viewport(
        self, screen_x: float, screen_y: float
    ) -> tuple[float, float]:
        """Convert screen coordinates to page viewport coordinates.

        Subtracts the browser window's on-screen position and the Chrome
        toolbar height (address bar + tabs, ~88px on macOS).
        """
        vx = screen_x - self._window_x
        vy = screen_y - self._window_y - self._toolbar_height
        return vx, vy

    def element_at_viewport(
        self, x: float, y: float
    ) -> Optional[GmailElement]:
        """Return the element whose bounding box contains viewport point (x, y).

        If multiple elements overlap, prefer smaller (more specific) elements.
        """
        candidates: list[GmailElement] = []
        for el in self._elements:
            b = el.bbox
            if b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]:
                candidates.append(el)
        if not candidates:
            return None
        # Prefer smallest area (most specific hit target)
        candidates.sort(key=lambda e: e.bbox["width"] * e.bbox["height"])
        return candidates[0]

    # ── Actions ────────────────────────────────────────────────────────────────

    async def click_element(self, element: GmailElement):
        """Click the center of a GmailElement by viewport coordinates."""
        if self._page is None:
            return
        b = element.bbox
        cx = b["x"] + b["width"] / 2
        cy = b["y"] + b["height"] / 2
        await self._page.mouse.click(cx, cy)
        await asyncio.sleep(0.3)

    async def archive_email(self):
        """Archive the currently selected / focused email.

        Works both from inbox (select + keyboard shortcut) and from within
        an open email thread.
        """
        if self._page is None:
            return
        # Keyboard shortcut 'e' archives in Gmail
        await self._page.keyboard.press("e")
        await asyncio.sleep(0.5)

    async def star_email(self, element: GmailElement):
        """Toggle the star on an email row element."""
        if self._page is None:
            return
        # Try to find the star button within the element's row
        b = element.bbox
        try:
            star = await self._page.evaluate(
                """
                ([x, y]) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    const row = el.closest('[role="row"], .zA, .zE');
                    if (!row) return null;
                    const star = row.querySelector(
                        '[data-tooltip*="Star"], [aria-label*="Star"], [aria-label*="Not starred"]'
                    );
                    if (!star) return null;
                    const r = star.getBoundingClientRect();
                    return {x: r.left + r.width / 2, y: r.top + r.height / 2};
                }
                """,
                [b["x"] + b["width"] / 2, b["y"] + b["height"] / 2],
            )
            if star:
                await self._page.mouse.click(star["x"], star["y"])
                await asyncio.sleep(0.3)
                return
        except Exception:
            pass
        # Fallback: click element then use keyboard shortcut 's'
        await self.click_element(element)
        await self._page.keyboard.press("s")
        await asyncio.sleep(0.3)

    async def reply_to_email(self, text: str):
        """Open the reply composer for the current email and type text."""
        if self._page is None:
            return
        # Keyboard shortcut 'r' opens reply in Gmail
        await self._page.keyboard.press("r")
        await asyncio.sleep(1.0)
        try:
            reply_box = await self._page.wait_for_selector(
                '[aria-label*="Message Body"], [g_editable="true"], .Am.Al.editable',
                timeout=5000,
            )
            await reply_box.click()
            await reply_box.type(text, delay=20)
        except Exception:
            # Fallback: just type if focus is already in composer
            await self._page.keyboard.type(text, delay=20)

    async def compose_email(self, to: str, subject: str, body: str):
        """Open the compose window and fill in To, Subject, and Body fields."""
        if self._page is None:
            return
        # Click compose button
        try:
            compose_btn = await self._page.wait_for_selector(
                '[gh="cm"], [aria-label="Compose"], [data-tooltip="Compose"]',
                timeout=5000,
            )
            await compose_btn.click()
        except Exception:
            # Keyboard shortcut 'c' opens compose
            await self._page.keyboard.press("c")
        await asyncio.sleep(1.0)

        # To field
        try:
            to_field = await self._page.wait_for_selector(
                'textarea[name="to"], input[aria-label*="To"], [name="to"]',
                timeout=5000,
            )
            await to_field.click()
            await to_field.type(to, delay=20)
            await self._page.keyboard.press("Tab")
        except Exception:
            pass

        # Subject field
        try:
            subj_field = await self._page.wait_for_selector(
                'input[name="subjectbox"], input[aria-label*="Subject"]',
                timeout=3000,
            )
            await subj_field.click()
            await subj_field.fill(subject)
        except Exception:
            pass

        # Body field
        try:
            body_field = await self._page.wait_for_selector(
                '[aria-label*="Message Body"], [g_editable="true"], .Am.Al.editable',
                timeout=3000,
            )
            await body_field.click()
            await body_field.type(body, delay=20)
        except Exception:
            pass

    async def scroll_inbox(self, direction: str = "down"):
        """Scroll the inbox list up or down.

        Args:
            direction: "up" or "down"
        """
        if self._page is None:
            return
        delta = 400 if direction == "down" else -400
        await self._page.mouse.wheel(0, delta)
        await asyncio.sleep(0.2)

    async def back_to_inbox(self):
        """Navigate back to the inbox view."""
        if self._page is None:
            return
        try:
            # Try back button first
            back = await self._page.query_selector(
                '[aria-label="Back to Inbox"], [aria-label*="Back"]'
            )
            if back:
                await back.click()
                await asyncio.sleep(0.5)
                return
        except Exception:
            pass
        # Keyboard shortcut 'u' goes back to thread list in Gmail
        await self._page.keyboard.press("u")
        await asyncio.sleep(0.5)

    async def get_email_body(self) -> str:
        """Return the full text content of the currently open email thread."""
        if self._page is None:
            return ""
        try:
            body = await self._page.evaluate("""
                () => {
                    const bodies = Array.from(
                        document.querySelectorAll('.a3s.aiL, .ii.gt div, .gs .ii')
                    );
                    return bodies.map(el => el.innerText.trim()).join('\\n\\n---\\n\\n');
                }
            """)
            return body or ""
        except Exception:
            return ""

    # ── Convenience helpers ────────────────────────────────────────────────────

    async def open_email_at(self, screen_x: float, screen_y: float):
        """Open the email row under gaze coordinates (screen space).

        Converts screen coords → viewport coords → element lookup → click.
        """
        vx, vy = self.screen_to_viewport(screen_x, screen_y)
        el = self.element_at_viewport(vx, vy)
        if el and el.type == "email_row":
            await self.click_element(el)

    async def gaze_click(self, screen_x: float, screen_y: float):
        """Click whatever element is under the given screen gaze coordinates."""
        vx, vy = self.screen_to_viewport(screen_x, screen_y)
        el = self.element_at_viewport(vx, vy)
        if el:
            await self.click_element(el)
        else:
            # Raw click at the viewport coordinate
            if self._page:
                await self._page.mouse.click(vx, vy)

    def get_cached_elements(self) -> list[GmailElement]:
        """Return the most recently scanned element list without re-scanning."""
        return self._elements


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

async def _smoke_test():
    ctrl = GmailController()
    await ctrl.start()
    print("Browser launched — waiting 3s for page load...")
    await asyncio.sleep(3)
    ctx = await ctrl.get_context()
    print(f"View: {ctx.view}")
    print(f"Elements found: {len(ctx.elements)}")
    print(f"Unread: {ctx.unread_count}")
    for el in ctx.elements[:10]:
        print(f"  [{el.type}] {el.label!r:60s}  bbox={el.bbox}")
    await ctrl.stop()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
