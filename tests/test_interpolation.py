import pytest

from cua.artifact.schema import InputParam, ParamType
from cua.replay.interpolation import InputValidationError, interpolate_steps, validate_inputs

from factories import build_artifact, valid_steps


def test_validate_inputs_accepts_valid_value():
    artifact = build_artifact()
    assert validate_inputs(artifact, {"member_id": "12345"}) == {"member_id": "12345"}


def test_validate_inputs_rejects_missing_required():
    artifact = build_artifact()
    with pytest.raises(InputValidationError):
        validate_inputs(artifact, {})


def test_validate_inputs_rejects_unexpected_key():
    artifact = build_artifact()
    with pytest.raises(InputValidationError):
        validate_inputs(artifact, {"member_id": "12345", "extra": "nope"})


def test_validate_inputs_coerces_int():
    artifact = build_artifact(inputs=[InputParam(name="member_id", type=ParamType.INT, description="id")])
    assert validate_inputs(artifact, {"member_id": "12345"}) == {"member_id": 12345}


def test_validate_inputs_rejects_uncoercible_value():
    artifact = build_artifact(inputs=[InputParam(name="member_id", type=ParamType.INT, description="id")])
    with pytest.raises(InputValidationError):
        validate_inputs(artifact, {"member_id": "not-a-number"})


def test_optional_input_may_be_omitted():
    artifact = build_artifact(
        inputs=[InputParam(name="member_id", type=ParamType.STRING, description="id", required=False)]
    )
    assert validate_inputs(artifact, {}) == {}


def test_interpolate_steps_substitutes_token():
    rendered = interpolate_steps(valid_steps(), {"member_id": "99999"})
    fill_step = next(s for s in rendered if s.step_id == "s2_fill")
    assert fill_step.params["value"] == "99999"


def test_interpolate_steps_does_not_mutate_original_steps():
    steps = valid_steps()
    interpolate_steps(steps, {"member_id": "99999"})
    fill_step = next(s for s in steps if s.step_id == "s2_fill")
    assert fill_step.params["value"] == "{{member_id}}"


def test_interpolate_steps_raises_on_missing_token_value():
    with pytest.raises(InputValidationError):
        interpolate_steps(valid_steps(), {})
