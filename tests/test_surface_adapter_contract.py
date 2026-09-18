import pytest

from cua.surfaces.base import SurfaceAdapter, TargetAmbiguous, TargetNotFound


def test_surface_adapter_is_abstract():
    with pytest.raises(TypeError):
        SurfaceAdapter()  # type: ignore[abstract]


def test_target_ambiguous_carries_context():
    exc = TargetAmbiguous(count=3, strategy="text")
    assert exc.count == 3
    assert exc.strategy == "text"
    assert "3" in str(exc)


def test_target_not_found_is_a_plain_exception():
    exc = TargetNotFound("no match for role_name, text, dom_selector")
    assert "no match" in str(exc)
