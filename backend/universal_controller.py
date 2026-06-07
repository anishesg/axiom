"""Universal Browser Controller — Playwright-based web automation for BCI gaze control.

Uses a persistent Chromium browser context so the user stays logged in.
Works on ANY website by combining the accessibility tree (semantic roles/names)
with DOM bounding boxes for spatial gaze-to-element mapping.

Browser window: positioned at (0, 0), viewport 1400x900.
Chrome toolbar height on macOS: ~88px (used for screen->viewport conversion).
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, BrowserContext, Page

logger = logging.getLogger(__name__)

USER_DATA_DIR = "/Users/anishkataria/axiom/data/chrome_profile"
WINDOW_X = 0
WINDOW_Y = 0
VIEWPORT_WIDTH = 1400
VIEWPORT_HEIGHT = 900
CHROME_TOOLBAR_HEIGHT = 88
SCROLL_DELTA = 400

INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "combobox", "menuitem",
    "tab", "radio", "slider", "switch", "searchbox", "option",
    "menuitemcheckbox", "menuitemradio", "spinbutton", "treeitem",
})

VISUAL_ROLES = frozenset({
    "heading", "img", "listitem", "cell", "row", "separator",
    "alert", "status", "tooltip", "banner", "navigation", "dialog",
    "progressbar", "meter",
})

ALL_RELEVANT_ROLES = INTERACTIVE_ROLES | VISUAL_ROLES

_SCAN_JS = """
(() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const sel = [
        '[role]', 'a[href]', 'button', 'input', 'textarea', 'select',
        '[tabindex]', '[onclick]', '[contenteditable="true"]',
        'label', 'summary', 'details',
    ].join(',');

    const seen = new Set();
    const results = [];

    for (const el of document.querySelectorAll(sel)) {
        if (seen.has(el)) continue;
        seen.add(el);

        const r = el.getBoundingClientRect();
        if (r.width <= 1 || r.height <= 1) continue;
        if (r.right < 0 || r.bottom < 0) continue;
        if (r.left > vw || r.top > vh) continue;

        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none') continue;
        if (style.opacity === '0') continue;

        const tag = el.tagName.toLowerCase();
        let role = el.getAttribute('role') || '';
        if (!role) {
            if (tag === 'a' && el.hasAttribute('href')) role = 'link';
            else if (tag === 'button') role = 'button';
            else if (tag === 'input') {
                const t = (el.getAttribute('type') || 'text').toLowerCase();
                if (t === 'checkbox') role = 'checkbox';
                else if (t === 'radio') role = 'radio';
                else if (t === 'submit' || t === 'button' || t === 'reset') role = 'button';
                else if (t === 'search') role = 'searchbox';
                else if (t === 'range') role = 'slider';
                else role = 'textbox';
            }
            else if (tag === 'textarea') role = 'textbox';
            else if (tag === 'select') role = 'combobox';
            else if (tag === 'summary') role = 'button';
            else if (el.getAttribute('contenteditable') === 'true') role = 'textbox';
            else if (el.getAttribute('tabindex') !== null || el.getAttribute('onclick') !== null) role = 'button';
        }

        const rawText = (el.innerText || el.textContent || '').trim();
        const name = el.getAttribute('aria-label')
            || el.getAttribute('title')
            || el.getAttribute('alt')
            || el.getAttribute('placeholder')
            || (tag === 'input' ? (el.getAttribute('value') || '') : '')
            || rawText.slice(0, 200);

        const meta = {};
        if (el.getAttribute('href')) meta.url = el.getAttribute('href');
        if (el.value !== undefined && el.value !== '') meta.value = String(el.value).slice(0, 500);
        if (el.checked !== undefined) meta.checked = el.checked;
        if (el.getAttribute('aria-expanded')) meta.expanded = el.getAttribute('aria-expanded') === 'true';
        if (el.getAttribute('aria-selected')) meta.selected = el.getAttribute('aria-selected') === 'true';
        if (el.getAttribute('aria-disabled')) meta.disabled = el.getAttribute('aria-disabled') === 'true';
        if (el.disabled) meta.disabled = true;

        const focused = document.activeElement === el;

        results.push({
            role: role,
            name: name.slice(0, 200),
            x: r.left,
            y: r.top,
            width: r.width,
            height: r.height,
            focused: focused,
            meta: meta,
        });
    }
    return JSON.stringify(results);
})()
"""


@dataclass
class WebElement:
    id: str
    role: str
    name: str
    bbox: dict
    level: int
    focused: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class PageContext:
    url: str
    title: str
    domain: str
    elements: list[WebElement]
    focused_element: Optional[WebElement]
    viewport_size: dict
    window_bounds: dict


class UniversalController:
    """Playwright-based universal browser controller for gaze/BCI interaction."""

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
        self._elements: list[WebElement] = []

    @property
    def page(self) -> Optional[Page]:
        return self._page

    async def start(self):
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

        self._browser.on("page", self._on_new_page)

    async def _on_new_page(self, page: Page):
        self._page = page
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass

    async def stop(self):
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

    async def _active_page(self) -> Optional[Page]:
        if self._browser is None:
            return None
        pages = self._browser.pages
        if not pages:
            return None
        if self._page is None or self._page.is_closed():
            self._page = pages[-1]
        return self._page

    async def scan_elements(self) -> list[WebElement]:
        page = await self._active_page()
        if page is None:
            return []

        try:
            raw_json: str = await page.evaluate(_SCAN_JS)
            raw_list: list[dict] = json.loads(raw_json)
        except Exception as e:
            logger.debug("scan_elements failed: %s", e)
            return []

        elements: list[WebElement] = []
        for raw in raw_list:
            role = raw.get("role", "")
            if not role:
                continue

            name = raw.get("name", "")
            bbox = {
                "x": raw["x"],
                "y": raw["y"],
                "width": raw["width"],
                "height": raw["height"],
            }

            el = WebElement(
                id=uuid.uuid4().hex[:12],
                role=role,
                name=name,
                bbox=bbox,
                level=0,
                focused=raw.get("focused", False),
                metadata=raw.get("meta", {}),
            )
            elements.append(el)

        # Enrich with accessibility tree depth info when available
        try:
            snapshot = await page.accessibility.snapshot()
            if snapshot:
                self._apply_tree_levels(snapshot, elements)
        except Exception:
            pass

        self._elements = elements
        return elements

    def _apply_tree_levels(self, tree: dict, elements: list[WebElement]):
        name_role_to_level: dict[tuple[str, str], int] = {}
        self._walk_tree(tree, 0, name_role_to_level)

        for el in elements:
            key = (el.name.strip(), el.role)
            if key in name_role_to_level and key[0]:
                el.level = name_role_to_level[key]

    def _walk_tree(
        self,
        node: dict,
        depth: int,
        out: dict[tuple[str, str], int],
    ):
        name = (node.get("name") or "").strip()
        role = node.get("role", "")
        if name and role:
            key = (name, role)
            if key not in out:
                out[key] = depth

        for child in node.get("children", []):
            self._walk_tree(child, depth + 1, out)

    async def get_context(self) -> PageContext:
        page = await self._active_page()
        elements = await self.scan_elements()

        url = page.url if page else ""
        title = ""
        if page:
            try:
                title = await page.title()
            except Exception:
                pass

        domain = ""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
        except Exception:
            pass

        focused = None
        for el in elements:
            if el.focused:
                focused = el
                break

        return PageContext(
            url=url,
            title=title,
            domain=domain,
            elements=elements,
            focused_element=focused,
            viewport_size={
                "width": self._viewport_width,
                "height": self._viewport_height,
            },
            window_bounds={
                "x": self._window_x,
                "y": self._window_y,
                "width": self._viewport_width,
                "height": self._viewport_height + self._toolbar_height,
            },
        )

    def screen_to_viewport(
        self, screen_x: float, screen_y: float
    ) -> tuple[float, float]:
        vx = screen_x - self._window_x
        vy = screen_y - self._window_y - self._toolbar_height
        return vx, vy

    def element_at_viewport(
        self, x: float, y: float
    ) -> Optional[WebElement]:
        candidates: list[WebElement] = []
        for el in self._elements:
            b = el.bbox
            if (b["x"] <= x <= b["x"] + b["width"]
                    and b["y"] <= y <= b["y"] + b["height"]):
                candidates.append(el)
        if not candidates:
            return None
        candidates.sort(key=lambda e: e.bbox["width"] * e.bbox["height"])
        return candidates[0]

    # ── Actions ────────────────────────────────────────────────────────────────

    async def click_element(self, element: WebElement):
        page = await self._active_page()
        if page is None:
            return
        b = element.bbox
        cx = b["x"] + b["width"] / 2
        cy = b["y"] + b["height"] / 2
        await page.mouse.click(cx, cy)
        await asyncio.sleep(0.3)

    async def scroll(self, direction: str = "down"):
        page = await self._active_page()
        if page is None:
            return
        dx, dy = 0, 0
        if direction == "down":
            dy = SCROLL_DELTA
        elif direction == "up":
            dy = -SCROLL_DELTA
        elif direction == "right":
            dx = SCROLL_DELTA
        elif direction == "left":
            dx = -SCROLL_DELTA
        await page.mouse.wheel(dx, dy)
        await asyncio.sleep(0.15)

    async def navigate_back(self):
        page = await self._active_page()
        if page is None:
            return
        await page.go_back()
        await asyncio.sleep(0.5)

    async def navigate_forward(self):
        page = await self._active_page()
        if page is None:
            return
        await page.go_forward()
        await asyncio.sleep(0.5)

    async def navigate_to(self, url: str):
        page = await self._active_page()
        if page is None:
            return
        if not url.startswith(("http://", "https://", "about:", "chrome:")):
            url = "https://" + url
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)

    async def type_text(self, element: WebElement, text: str, clear: bool = False):
        page = await self._active_page()
        if page is None:
            return
        b = element.bbox
        cx = b["x"] + b["width"] / 2
        cy = b["y"] + b["height"] / 2
        await page.mouse.click(cx, cy)
        await asyncio.sleep(0.1)
        if clear:
            await page.keyboard.press("Meta+a")
            await asyncio.sleep(0.05)
        await page.keyboard.type(text, delay=15)

    async def press_key(self, key: str):
        page = await self._active_page()
        if page is None:
            return
        await page.keyboard.press(key)

    async def open_in_new_tab(self, element: WebElement):
        page = await self._active_page()
        if page is None:
            return
        b = element.bbox
        cx = b["x"] + b["width"] / 2
        cy = b["y"] + b["height"] / 2
        await page.keyboard.down("Meta")
        await page.mouse.click(cx, cy)
        await page.keyboard.up("Meta")
        await asyncio.sleep(0.5)

    async def close_tab(self):
        page = await self._active_page()
        if page is None:
            return
        if self._browser and len(self._browser.pages) <= 1:
            return
        await page.close()
        if self._browser and self._browser.pages:
            self._page = self._browser.pages[-1]
            try:
                await self._page.bring_to_front()
            except Exception:
                pass

    async def switch_tab(self, index: int):
        if self._browser is None:
            return
        pages = self._browser.pages
        if 0 <= index < len(pages):
            self._page = pages[index]
            try:
                await self._page.bring_to_front()
            except Exception:
                pass

    async def get_tabs(self) -> list[dict]:
        if self._browser is None:
            return []
        tabs = []
        for i, p in enumerate(self._browser.pages):
            title = ""
            try:
                title = await p.title()
            except Exception:
                pass
            tabs.append({
                "index": i,
                "title": title,
                "url": p.url,
                "active": p == self._page,
            })
        return tabs

    async def get_page_text(self) -> str:
        page = await self._active_page()
        if page is None:
            return ""
        try:
            text = await page.evaluate("""
                () => {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: (node) => {
                                const el = node.parentElement;
                                if (!el) return NodeFilter.FILTER_REJECT;
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden')
                                    return NodeFilter.FILTER_REJECT;
                                const tag = el.tagName.toLowerCase();
                                if (tag === 'script' || tag === 'style' || tag === 'noscript')
                                    return NodeFilter.FILTER_REJECT;
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );
                    const chunks = [];
                    let total = 0;
                    while (walker.nextNode() && total < 10000) {
                        const t = walker.currentNode.textContent.trim();
                        if (t.length > 0) {
                            chunks.push(t);
                            total += t.length;
                        }
                    }
                    return chunks.join('\\n');
                }
            """)
            return text or ""
        except Exception:
            return ""

    # ── Gaze highlight ────────────────────────────────────────────────────────

    async def highlight_element(self, element: WebElement | None, commitment: float = 0.0):
        """Inject a visual highlight on the page around the gazed element."""
        page = await self._active_page()
        if page is None:
            return
        try:
            if element is None:
                await page.evaluate("document.getElementById('_axiom_hl')?.remove()")
                return
            b = element.bbox
            await page.evaluate("""([x, y, w, h, commit, role, name]) => {
                let el = document.getElementById('_axiom_hl');
                if (!el) {
                    el = document.createElement('div');
                    el.id = '_axiom_hl';
                    el.style.cssText = 'position:fixed;pointer-events:none;z-index:999999;border-radius:6px;transition:all 0.15s ease;box-sizing:border-box;';
                    document.body.appendChild(el);
                    const lbl = document.createElement('div');
                    lbl.id = '_axiom_lbl';
                    lbl.style.cssText = 'position:fixed;pointer-events:none;z-index:999999;font:bold 11px monospace;padding:2px 8px;border-radius:3px;white-space:nowrap;transition:all 0.15s ease;';
                    document.body.appendChild(lbl);
                }
                const lbl = document.getElementById('_axiom_lbl');
                const pad = 4;
                el.style.left = (x - pad) + 'px';
                el.style.top = (y - pad) + 'px';
                el.style.width = (w + pad * 2) + 'px';
                el.style.height = (h + pad * 2) + 'px';
                const alpha = 0.25 + commit * 0.6;
                const glow = commit > 0.7 ? '0 0 20px rgba(0,255,136,0.5)' : commit > 0.4 ? '0 0 12px rgba(0,212,255,0.3)' : 'none';
                const color = commit > 0.7 ? 'rgba(0,255,136,' : commit > 0.4 ? 'rgba(0,212,255,' : 'rgba(108,99,255,';
                el.style.border = '2px solid ' + color + Math.min(1, alpha + 0.3) + ')';
                el.style.background = color + (alpha * 0.15) + ')';
                el.style.boxShadow = glow;
                lbl.style.left = (x - pad) + 'px';
                lbl.style.top = (y - pad - 20) + 'px';
                lbl.style.background = color + '0.9)';
                lbl.style.color = '#fff';
                lbl.textContent = role.toUpperCase() + ' · ' + (name || '').slice(0, 50);
            }""", [b["x"], b["y"], b["width"], b["height"], commitment,
                   element.role, element.name])
        except Exception:
            pass

    # ── Gaze convenience ───────────────────────────────────────────────────────

    async def gaze_click(self, screen_x: float, screen_y: float):
        vx, vy = self.screen_to_viewport(screen_x, screen_y)
        el = self.element_at_viewport(vx, vy)
        if el:
            await self.click_element(el)
        else:
            page = await self._active_page()
            if page:
                await page.mouse.click(vx, vy)

    def get_cached_elements(self) -> list[WebElement]:
        return self._elements

    def interactive_elements(self) -> list[WebElement]:
        return [el for el in self._elements if el.role in INTERACTIVE_ROLES]


async def _smoke_test():
    ctrl = UniversalController()
    await ctrl.start()
    print("Browser launched — waiting 2s...")
    await asyncio.sleep(2)

    await ctrl.navigate_to("https://news.ycombinator.com")
    await asyncio.sleep(2)

    ctx = await ctrl.get_context()
    print(f"URL:    {ctx.url}")
    print(f"Title:  {ctx.title}")
    print(f"Domain: {ctx.domain}")
    print(f"Elements: {len(ctx.elements)} total, "
          f"{len(ctrl.interactive_elements())} interactive")
    if ctx.focused_element:
        print(f"Focused: [{ctx.focused_element.role}] {ctx.focused_element.name!r}")

    for el in ctx.elements[:20]:
        area = el.bbox['width'] * el.bbox['height']
        print(f"  [{el.role:12s}] lv{el.level} {el.name!r:50s} "
              f"{el.bbox['width']:.0f}x{el.bbox['height']:.0f} ({area:.0f}px²)")

    tabs = await ctrl.get_tabs()
    print(f"\nTabs ({len(tabs)}):")
    for t in tabs:
        marker = " *" if t["active"] else ""
        print(f"  [{t['index']}] {t['title']}{marker}")

    text = await ctrl.get_page_text()
    print(f"\nPage text ({len(text)} chars): {text[:200]}...")

    await ctrl.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
