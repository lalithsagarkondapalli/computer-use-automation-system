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
