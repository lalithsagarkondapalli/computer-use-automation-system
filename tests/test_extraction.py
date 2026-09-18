import pytest

from cua.artifact.extraction import ExtractionSpec
from cua.artifact.targets import TargetDescriptor
from cua.replay.extraction import ExtractionError, extract_output
from cua.surfaces.base import TargetNotFound

from fakes import FakeElement, FakeSurfaceAdapter


async def test_extract_text_no_transform():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    spec = ExtractionSpec(output_name="savings_balance", target=TargetDescriptor(role="status", accessible_name="Savings Balance"))
    assert await extract_output(surface, spec) == "$1,234.56"


async def test_extract_currency_to_float():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    spec = ExtractionSpec(
        output_name="savings_balance",
        target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
        transform="currency_to_float",
    )
    assert await extract_output(surface, spec) == 1234.56


async def test_extract_strip_transform():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("status", "Note", [FakeElement(text="  hello  ")])
    spec = ExtractionSpec(output_name="note", target=TargetDescriptor(role="status", accessible_name="Note"), transform="strip")
    assert await extract_output(surface, spec) == "hello"


async def test_currency_to_float_rejects_unparseable_value():
    surface = FakeSurfaceAdapter()
    surface.register_role_name("status", "Note", [FakeElement(text="not-a-number")])
    spec = ExtractionSpec(
        output_name="note", target=TargetDescriptor(role="status", accessible_name="Note"), transform="currency_to_float"
    )
    with pytest.raises(ExtractionError):
        await extract_output(surface, spec)


async def test_extract_propagates_target_not_found():
    surface = FakeSurfaceAdapter()
    spec = ExtractionSpec(output_name="x", target=TargetDescriptor(dom_selector="#nope"))
    with pytest.raises(TargetNotFound):
        await extract_output(surface, spec)
