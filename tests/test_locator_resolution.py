import pytest

from cua.artifact.targets import TargetDescriptor
from cua.surfaces.base import TargetAmbiguous, TargetNotFound

from fakes import FakeElement, FakeSurfaceAdapter


async def test_first_matching_strategy_wins():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    resolved = await surface.resolve(TargetDescriptor(role="textbox", accessible_name="Member ID"))
    assert resolved.strategy == "role_name"


async def test_label_strategy_resolution():
    surface = FakeSurfaceAdapter()
    surface.register_label("Member ID", [FakeElement()])
    resolved = await surface.resolve(TargetDescriptor(nearby_anchor="Member ID"))
    assert resolved.strategy == "label"


async def test_zero_matches_falls_through_to_next_strategy():
    surface = FakeSurfaceAdapter()
    surface.register_dom_selector("#fallback", [FakeElement(text="ok")])
    target = TargetDescriptor(role="button", accessible_name="Missing", dom_selector="#fallback")
    resolved = await surface.resolve(target)
    assert resolved.strategy == "dom_selector"


async def test_multiple_matches_without_disambiguation_is_ambiguous():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Duplicate", [FakeElement(), FakeElement()])
    with pytest.raises(TargetAmbiguous) as exc_info:
        await surface.resolve(TargetDescriptor(role="button", accessible_name="Duplicate"))
    assert exc_info.value.count == 2
    assert exc_info.value.strategy == "role_name"


async def test_visible_text_disambiguates_multiple_matches():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Action", [FakeElement(text="Save Draft"), FakeElement(text="Save Final")])
    resolved = await surface.resolve(TargetDescriptor(role="button", accessible_name="Action", visible_text="Save Final"))
    assert resolved.strategy == "role_name+text_filter"
    assert resolved.handle.text == "Save Final"


async def test_no_strategy_matches_raises_not_found():
    surface = FakeSurfaceAdapter()
    with pytest.raises(TargetNotFound):
        await surface.resolve(TargetDescriptor(dom_selector="#nope"))


async def test_strategy_order_is_respected_even_when_a_later_strategy_would_also_match():
    surface = FakeSurfaceAdapter()
    # Both role_name and dom_selector would match -- role_name comes first
    # in strategy_order and must win, proving order (not "any match") governs.
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.register_dom_selector("#search-btn", [FakeElement()])
    resolved = await surface.resolve(TargetDescriptor(role="button", accessible_name="Search", dom_selector="#search-btn"))
    assert resolved.strategy == "role_name"
