"""Integration tests against a real, headless Chromium browser via
Playwright -- the tests that actually exercise PlaywrightWebSurface's
Playwright-specific code. Everything else in this suite uses
FakeSurfaceAdapter for speed; these prove the real implementation behaves
the same way against an actual browser (see fixtures/sample_page.html).

Requires the `playwright` package and a Chromium binary. See README.md
"Browser setup" for the exact command. If `PLAYWRIGHT_EXECUTABLE_PATH` is
set, it is passed through to `PlaywrightWebSurface.launch()` -- this
sandbox pins a pre-installed Chromium at a fixed path rather than
downloading a new one (see the environment notes in the review package).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from cua.artifact.checks import ElementNotVisibleCheck, ElementVisibleCheck, UrlMatchesCheck
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.base import TargetAmbiguous, TargetNotFound
from cua.surfaces.playwright_web import PlaywrightWebSurface

# Loaded via a file:// URL rather than a data: URL -- see "Bugs discovered
# during implementation" in the Phase 3 review package. Chromium does not
# treat a `location.hash` mutation on a data: URL page as a navigation (the
# page's URL and Playwright's navigation events never update), so the
# fixture's own in-page `location.hash='#results'` click handler -- exercised
# by test_click_reveals_results_and_url_checkpoint_passes below -- is
# unobservable there. The same handler, and the same wait_for_checkpoint()
# code path, behave correctly against file:// (verified with a standalone
# script before making this change). This is a browser/URL-scheme behavior,
# not a bug in PlaywrightWebSurface or CheckpointEngine.
FIXTURE_PATH = (Path(__file__).parent / "fixtures" / "sample_page.html").resolve()
_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")


async def _open_fixture() -> PlaywrightWebSurface:
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    await surface.navigate(f"file://{FIXTURE_PATH}")
    return surface


async def test_role_name_resolution_and_fill():
    async with await _open_fixture() as surface:
        resolved = await surface.resolve(TargetDescriptor(role="textbox", accessible_name="Member ID"))
        await surface.fill(resolved, "12345")
        assert (await surface.read(resolved, "value")) == "12345"


async def test_dom_selector_fallback():
    async with await _open_fixture() as surface:
        resolved = await surface.resolve(TargetDescriptor(dom_selector="#legacy-fallback"))
        assert (await surface.read(resolved, "text")) == "Legacy Control"


async def test_ambiguous_control_raises_without_disambiguation():
    async with await _open_fixture() as surface:
        with pytest.raises(TargetAmbiguous) as exc_info:
            await surface.resolve(TargetDescriptor(role="button", accessible_name="Duplicate"))
        assert exc_info.value.count == 2


async def test_visible_text_disambiguates_same_role_and_name():
    async with await _open_fixture() as surface:
        resolved = await surface.resolve(TargetDescriptor(role="button", accessible_name="Action", visible_text="Save Final"))
        assert resolved.strategy == "role_name+text_filter"


async def test_not_found_when_no_strategy_matches():
    async with await _open_fixture() as surface:
        with pytest.raises(TargetNotFound):
            await surface.resolve(TargetDescriptor(dom_selector="#does-not-exist"))


async def test_click_reveals_results_and_url_checkpoint_passes():
    async with await _open_fixture() as surface:
        search = await surface.resolve(TargetDescriptor(role="button", accessible_name="Search"))
        await surface.click(search)
        assert await surface.wait_for_checkpoint(UrlMatchesCheck(pattern="#results"), timeout_ms=2000) is True


async def test_element_visible_checkpoint_waits_for_the_click_before_passing():
    async with await _open_fixture() as surface:
        balance = TargetDescriptor(role="status", accessible_name="Savings Balance")
        assert await surface.wait_for_checkpoint(ElementVisibleCheck(target=balance), timeout_ms=300) is False

        search = await surface.resolve(TargetDescriptor(role="button", accessible_name="Search"))
        await surface.click(search)

        assert await surface.wait_for_checkpoint(ElementVisibleCheck(target=balance), timeout_ms=2000) is True
        resolved = await surface.resolve(balance)
        assert (await surface.read(resolved, "text")) == "$1,234.56"


async def test_element_not_visible_checkpoint_after_click_hides_the_spinner():
    async with await _open_fixture() as surface:
        spinner = TargetDescriptor(role="status", accessible_name="Loading")
        search = await surface.resolve(TargetDescriptor(role="button", accessible_name="Search"))
        await surface.click(search)
        assert await surface.wait_for_checkpoint(ElementNotVisibleCheck(target=spinner), timeout_ms=2000) is True


async def test_screenshot_returns_nonempty_bytes():
    async with await _open_fixture() as surface:
        data = await surface.screenshot()
        assert isinstance(data, bytes)
        assert len(data) > 0
