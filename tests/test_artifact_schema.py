import pytest
from pydantic import ValidationError

from cua.actions import ActionType
from cua.artifact.schema import CapabilityArtifact, Step
from cua.artifact.targets import TargetDescriptor

from factories import base_kwargs, valid_steps


def test_valid_artifact_round_trips():
    artifact = CapabilityArtifact(**base_kwargs())
    assert artifact.schema_version == "1.0"
    assert len(artifact.steps) == 5


def test_unsupported_schema_version_rejected():
    kwargs = base_kwargs()
    kwargs["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        CapabilityArtifact(**kwargs)


def test_undeclared_interpolation_token_rejected():
    steps = valid_steps()
    steps[1].params["value"] = "{{not_a_declared_input}}"
    with pytest.raises(ValidationError):
        CapabilityArtifact(**base_kwargs(steps))


def test_declared_interpolation_token_is_fine():
    steps = valid_steps()
    steps[1].params["value"] = "{{member_id}}"
    CapabilityArtifact(**base_kwargs(steps))  # should not raise


def test_requires_at_least_one_checkpoint():
    steps = [s for s in valid_steps() if s.action != ActionType.CHECKPOINT]
    with pytest.raises(ValidationError):
        CapabilityArtifact(**base_kwargs(steps))


def test_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        CapabilityArtifact(**base_kwargs(steps=[]))


def test_duplicate_step_ids_rejected():
    steps = valid_steps()
    steps[1].step_id = steps[0].step_id
    with pytest.raises(ValidationError):
        CapabilityArtifact(**base_kwargs(steps))


def test_fill_without_value_param_rejected():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.FILL, target=TargetDescriptor(dom_selector="#x"), params={})


def test_click_without_target_rejected():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.CLICK, params={})


def test_navigate_without_route_or_url_rejected():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.NAVIGATE, params={})


def test_checkpoint_without_spec_rejected():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.CHECKPOINT)


def test_read_extract_without_spec_rejected():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.READ_EXTRACT, target=TargetDescriptor(dom_selector="#x"))


def test_finish_not_allowed_as_persisted_step():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.FINISH)


def test_request_human_not_allowed_as_persisted_step():
    with pytest.raises(ValidationError):
        Step(step_id="s", action=ActionType.REQUEST_HUMAN)


def test_retry_policy_bounded():
    with pytest.raises(ValidationError):
        Step(
            step_id="s",
            action=ActionType.WAIT,
            retry={"max_attempts": 100},  # far above the ge=1, le=5 bound
        )
