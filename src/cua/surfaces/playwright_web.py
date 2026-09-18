"""PlaywrightWebSurface: the concrete SurfaceAdapter for web applications.

This is the ONE place in the repository that imports Playwright. Nothing
in `cua.replay`, `cua.artifact`, `cua.safety`, or `cua.agent` depends on
this module -- they depend only on the `SurfaceAdapter` interface it
implements, which is what lets the same recorded-capability concept
extend, in design, to a legacy web app or a native desktop application
without touching those layers (see REPORT.md Section 4). Swapping this
module for a different one is the entire cost of adding a new surface.
"""
from __future__ import annotations

import asyncio

from playwright.async_api import Browser, BrowserContext, Page, Playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from cua.artifact.checks import CheckpointAssertion, ElementNotVisibleCheck, ElementVisibleCheck, TextPresentCheck, UrlMatchesCheck
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.base import ResolvedTarget, SurfaceAdapter, TargetAmbiguous, TargetNotFound
from cua.surfaces.observation import InteractiveElement, Observation

_POLL_INTERVAL_S = 0.1
_OBSERVE_SELECTOR = "button, a, input, select, [role]"
_OBSERVE_MAX_ELEMENTS = 50


class PlaywrightWebSurface(SurfaceAdapter):
    """One Playwright Page per instance.

    Construct via `PlaywrightWebSurface.launch()`; the caller owns the
    resulting instance's lifecycle and must call `close()` (or use it as
    an async context manager) when done -- see `cua.replay.engine`'s
    module docstring for why session lifecycle lives with the caller.
    """

    def __init__(self, playwright: Playwright, browser: Browser, context: BrowserContext, page: Page):
        self._playwright = playwright
        self._browser = browser
        self._context = context
        self._page = page

    @classmethod
    async def launch(cls, *, headed: bool = False, executable_path: str | None = None) -> "PlaywrightWebSurface":
        """`executable_path` overrides Playwright's normal browser
        resolution. Leave it unset in a normal environment (one where
        `playwright install` has downloaded a matching browser revision);
        it exists so this same code also runs in environments that pin a
        pre-installed Chromium at a fixed path (see README.md "Browser
        setup").
        """
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=not headed, executable_path=executable_path)
        context = await browser.new_context()
        page = await context.new_page()
        return cls(playwright, browser, context, page)

    async def __aenter__(self) -> "PlaywrightWebSurface":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def navigate(self, url_or_route: str) -> None:
        await self._page.goto(url_or_route, wait_until="load")

    async def resolve(self, target: TargetDescriptor) -> ResolvedTarget:
        for strategy in target.strategy_order:
            locator = self._build_locator(target, strategy)
            if locator is None:
                continue  # descriptor doesn't carry the signal this strategy needs
            count = await locator.count()
            if count == 0:
                continue
            if count == 1:
                return ResolvedTarget(handle=locator.first, strategy=strategy)
            # count > 1: try one deterministic disambiguation signal before
            # giving up -- never .first() a multi-match (Section 8).
            if target.visible_text and strategy != "text":
                narrowed = locator.filter(has_text=target.visible_text)
                if await narrowed.count() == 1:
                    return ResolvedTarget(handle=narrowed.first, strategy=f"{strategy}+text_filter")
            raise TargetAmbiguous(count=count, strategy=strategy)
        raise TargetNotFound(
            f"no control matched any strategy in {target.strategy_order} "
            f"(role={target.role!r}, accessible_name={target.accessible_name!r}, "
            f"nearby_anchor={target.nearby_anchor!r}, visible_text={target.visible_text!r}, "
            f"dom_selector={target.dom_selector!r})"
        )

    def _build_locator(self, target: TargetDescriptor, strategy: str):
        frame = self._page
        for frame_selector in target.frame_path:
            frame = frame.frame_locator(frame_selector)
        if strategy == "role_name":
            if not target.role or not target.accessible_name:
                return None
            return frame.get_by_role(target.role, name=target.accessible_name, exact=True)
        if strategy == "label":
            if not target.nearby_anchor:
                return None
            return frame.get_by_label(target.nearby_anchor, exact=True)
        if strategy == "text":
            if not target.visible_text:
                return None
            return frame.get_by_text(target.visible_text, exact=True)
        if strategy == "dom_selector":
            if not target.dom_selector:
                return None
            return frame.locator(target.dom_selector)
        raise ValueError(f"unknown locator strategy {strategy!r}")  # pragma: no cover -- strategy_order is validated upstream

    async def click(self, resolved: ResolvedTarget) -> None:
        await resolved.handle.click()

    async def fill(self, resolved: ResolvedTarget, value: str) -> None:
        await resolved.handle.fill(value)

    async def select(self, resolved: ResolvedTarget, value: str) -> None:
        await resolved.handle.select_option(value)

    async def read(self, resolved: ResolvedTarget, attribute: str) -> str:
        if attribute == "text":
            return (await resolved.handle.text_content()) or ""
        if attribute == "value":
            return await resolved.handle.input_value()
        if attribute == "href":
            return (await resolved.handle.get_attribute("href")) or ""
        if attribute == "aria_label":
            return (await resolved.handle.get_attribute("aria-label")) or ""
        raise ValueError(f"unknown read attribute {attribute!r}")  # pragma: no cover -- ExtractionSpec.attribute is a closed enum

    async def wait_for(self, target: TargetDescriptor, timeout_ms: int) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        while True:
            try:
                await self.resolve(target)
                return True
            except TargetNotFound:
                if asyncio.get_event_loop().time() >= deadline:
                    return False
                await asyncio.sleep(_POLL_INTERVAL_S)
            # TargetAmbiguous is not caught here -- it propagates immediately.
            # Ambiguity is a classification of its own, not a "not yet" state,
            # and swallowing it into a timeout would hide a real problem.

    async def wait_for_checkpoint(self, assertion: CheckpointAssertion, timeout_ms: int) -> bool:
        try:
            if isinstance(assertion, UrlMatchesCheck):
                await self._page.wait_for_url(f"**{assertion.pattern}**", timeout=timeout_ms)
                return True
            if isinstance(assertion, TextPresentCheck):
                await self._page.get_by_text(assertion.text).first.wait_for(state="visible", timeout=timeout_ms)
                return True
            if isinstance(assertion, ElementVisibleCheck):
                return await self._poll_visibility(assertion.target, timeout_ms, want_visible=True)
            if isinstance(assertion, ElementNotVisibleCheck):
                return await self._poll_visibility(assertion.target, timeout_ms, want_visible=False)
        except PlaywrightTimeoutError:
            return False
        raise ValueError(f"unhandled checkpoint assertion type: {type(assertion)!r}")  # pragma: no cover -- closed union

    async def _poll_visibility(self, target: TargetDescriptor, timeout_ms: int, *, want_visible: bool) -> bool:
        """ElementVisible/NotVisible checks poll rather than use a single
        native Playwright wait, because resolving `target` may itself
        require our deterministic multi-strategy, ambiguity-aware logic
        (Section 8) -- there is no single native primitive for that. The
        poll interval is short and the loop checks real state every tick,
        which is a materially different thing from a single blind
        `sleep()` before assuming success (Section 9/23).
        """
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        while True:
            try:
                resolved = await self.resolve(target)
                visible = await resolved.handle.is_visible()
            except TargetNotFound:
                visible = False
            if visible == want_visible:
                return True
            if asyncio.get_event_loop().time() >= deadline:
                return False
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def current_url(self) -> str:
        return self._page.url

    async def observe(self, step_number: int) -> Observation:
        elements = await self._page.eval_on_selector_all(
            _OBSERVE_SELECTOR,
            f"""(nodes) => nodes.slice(0, {_OBSERVE_MAX_ELEMENTS}).map(n => ({{
                role: n.getAttribute('role') || n.tagName.toLowerCase(),
                accessible_name: n.getAttribute('aria-label') || n.innerText || n.getAttribute('placeholder') || null,
                text: n.innerText || null,
                control_type: n.tagName.toLowerCase(),
            }}))""",
        )
        return Observation(
            url=self._page.url,
            title=await self._page.title(),
            interactive_elements=[InteractiveElement(**element) for element in elements],
            visible_text_summary=(await self._page.inner_text("body"))[:2000],
            step_number=step_number,
        )

    async def screenshot(self) -> bytes:
        return await self._page.screenshot()

    async def close(self) -> None:
        await self._context.close()
        await self._browser.close()
        await self._playwright.stop()
