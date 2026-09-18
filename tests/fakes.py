"""A minimal, deterministic in-memory SurfaceAdapter used by ReplayEngine
unit tests.

Lets tests exercise policy, retry, checkpoint, and extraction logic
without a real browser. Its `resolve()` implements the same strategy-
ordering algorithm as `PlaywrightWebSurface.resolve()` (independently,
not by sharing code -- see the Phase 3 review notes for why duplication
was chosen over a forced shared abstraction), so `test_locator_resolution.py`
proves the algorithm fast and `test_playwright_web_surface.py` proves the
real implementation matches it against an actual browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cua.artifact.checks import CheckpointAssertion
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.base import ResolvedTarget, SurfaceAdapter, TargetAmbiguous, TargetNotFound
from cua.surfaces.observation import Observation


@dataclass
class FakeElement:
    text: str = ""
    value: str = ""
    visible: bool = True

    async def is_visible(self) -> bool:
        return self.visible


class FakeSurfaceAdapter(SurfaceAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._by_role_name: dict[tuple[str, str], list[FakeElement]] = {}
        self._by_label: dict[str, list[FakeElement]] = {}
        self._by_text: dict[str, list[FakeElement]] = {}
        self._by_dom_selector: dict[str, list[FakeElement]] = {}
        self._flaky_role_name: dict[tuple[str, str], int] = {}
        self._checkpoint_results: dict[str, bool] = {}
        self._redirect_on_next_click: str | None = None
        self._url = "http://127.0.0.1:8000/"
        self.closed = False

    # -- test setup -----------------------------------------------------
    def register_role_name(self, role: str, name: str, elements: list[FakeElement]) -> None:
        self._by_role_name[(role, name)] = elements

    def register_role_name_flaky(self, role: str, name: str, elements: list[FakeElement], fail_times: int) -> None:
        self._by_role_name[(role, name)] = elements
        self._flaky_role_name[(role, name)] = fail_times

    def register_label(self, anchor: str, elements: list[FakeElement]) -> None:
        self._by_label[anchor] = elements

    def register_text(self, text: str, elements: list[FakeElement]) -> None:
        self._by_text[text] = elements

    def register_dom_selector(self, selector: str, elements: list[FakeElement]) -> None:
        self._by_dom_selector[selector] = elements

    def set_url(self, url: str) -> None:
        self._url = url

    def script_checkpoint(self, key: str, result: bool) -> None:
        self._checkpoint_results[key] = result

    def script_redirect_on_next_click(self, url: str) -> None:
        self._redirect_on_next_click = url

    # -- SurfaceAdapter ---------------------------------------------------
    async def navigate(self, url_or_route: str) -> None:
        self.calls.append(("navigate", url_or_route))
        if url_or_route.startswith("http://") or url_or_route.startswith("https://"):
            self._url = url_or_route
        else:
            scheme, _, rest = self._url.partition("://")
            host = rest.split("/", 1)[0]
            self._url = f"{scheme}://{host}{url_or_route}"

    async def resolve(self, target: TargetDescriptor) -> ResolvedTarget:
        for strategy in target.strategy_order:
            candidates = self._candidates_for(target, strategy)
            if candidates is None:
                continue
            if len(candidates) == 0:
                continue
            if len(candidates) == 1:
                return ResolvedTarget(handle=candidates[0], strategy=strategy)
            if target.visible_text and strategy != "text":
                narrowed = [c for c in candidates if c.text == target.visible_text]
                if len(narrowed) == 1:
                    return ResolvedTarget(handle=narrowed[0], strategy=f"{strategy}+text_filter")
            raise TargetAmbiguous(count=len(candidates), strategy=strategy)
        raise TargetNotFound(f"no fake element registered for {target!r}")

    def _candidates_for(self, target: TargetDescriptor, strategy: str) -> list[FakeElement] | None:
        if strategy == "role_name":
            if not target.role or not target.accessible_name:
                return None
            key = (target.role, target.accessible_name)
            remaining = self._flaky_role_name.get(key, 0)
            if remaining > 0:
                self._flaky_role_name[key] = remaining - 1
                return []
            return self._by_role_name.get(key, [])
        if strategy == "label":
            if not target.nearby_anchor:
                return None
            return self._by_label.get(target.nearby_anchor, [])
        if strategy == "text":
            if not target.visible_text:
                return None
            return self._by_text.get(target.visible_text, [])
        if strategy == "dom_selector":
            if not target.dom_selector:
                return None
            return self._by_dom_selector.get(target.dom_selector, [])
        return None

    async def click(self, resolved: ResolvedTarget) -> None:
        self.calls.append(("click", resolved.strategy))
        if self._redirect_on_next_click is not None:
            self._url = self._redirect_on_next_click
            self._redirect_on_next_click = None

    async def fill(self, resolved: ResolvedTarget, value: str) -> None:
        self.calls.append(("fill", resolved.strategy, value))
        resolved.handle.value = value

    async def select(self, resolved: ResolvedTarget, value: str) -> None:
        self.calls.append(("select", resolved.strategy, value))
        resolved.handle.value = value

    async def read(self, resolved: ResolvedTarget, attribute: str) -> str:
        self.calls.append(("read", resolved.strategy, attribute))
        element: FakeElement = resolved.handle
        if attribute == "value":
            return element.value
        return element.text

    async def wait_for(self, target: TargetDescriptor, timeout_ms: int) -> bool:
        try:
            await self.resolve(target)
            return True
        except TargetNotFound:
            return False

    async def wait_for_checkpoint(self, assertion: CheckpointAssertion, timeout_ms: int) -> bool:
        key = getattr(assertion, "pattern", None) or getattr(assertion, "text", None) or "unknown"
        return self._checkpoint_results.get(key, False)

    async def current_url(self) -> str:
        return self._url

    async def observe(self, step_number: int) -> Observation:
        raise NotImplementedError("FakeSurfaceAdapter is replay-only; discovery is exercised separately in Phase 4")

    async def screenshot(self) -> bytes:
        self.calls.append(("screenshot",))
        return b"fake-png-bytes"

    async def close(self) -> None:
        self.closed = True
