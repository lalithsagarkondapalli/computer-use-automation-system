import pytest
from pydantic import ValidationError

from cua.artifact.targets import TargetDescriptor


def test_requires_a_signal():
    with pytest.raises(ValidationError):
        TargetDescriptor()


def test_role_requires_accessible_name():
    with pytest.raises(ValidationError):
        TargetDescriptor(role="button")


def test_valid_role_and_name():
    t = TargetDescriptor(role="button", accessible_name="Search")
    assert t.strategy_order[0] == "role_name"


def test_dom_selector_alone_is_a_valid_fallback():
    t = TargetDescriptor(dom_selector="#submit-btn")
    assert t.role is None


def test_text_only_is_valid():
    TargetDescriptor(visible_text="Savings Balance")


def test_label_anchor_alone_is_a_valid_signal():
    # Regression test (Phase 3): once "label" became a real resolution
    # strategy (SurfaceAdapter.get_by_label(nearby_anchor)), a descriptor
    # carrying only a label anchor must be accepted -- the original
    # Phase 2 validator predated that strategy and rejected this case.
    t = TargetDescriptor(nearby_anchor="Member ID")
    assert t.role is None
    assert t.strategy_order[1] == "label"


def test_label_anchor_still_requires_a_nonempty_value():
    # The fix must not loosen validation into accepting an empty/blank
    # anchor as a "signal" -- a target with nothing meaningful set is
    # still rejected, same as before.
    with pytest.raises(ValidationError):
        TargetDescriptor(nearby_anchor=None, role=None, accessible_name=None, visible_text=None, dom_selector=None)
