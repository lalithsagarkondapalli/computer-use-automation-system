import pytest
from pydantic import ValidationError

from cua.artifact.checks import CheckpointSpec, ConditionProbe, TextPresentCheck, UrlMatchesCheck
from cua.replay.checkpoints import CheckpointEngine
from cua.results.taxonomy import OutcomeCode, RunStatus

from fakes import FakeSurfaceAdapter


async def test_checkpoint_passes_when_assertion_satisfied():
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("/results", True)
    checkpoint = CheckpointSpec(description="reached results", assertion=UrlMatchesCheck(pattern="/results"))
    outcome = await CheckpointEngine().verify(surface, checkpoint, timeout_ms=1000)
    assert outcome.passed is True


async def test_checkpoint_failure_classified_as_hard_failure():
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("/results", False)
    checkpoint = CheckpointSpec(description="reached results", assertion=UrlMatchesCheck(pattern="/results"))
    outcome = await CheckpointEngine().verify(surface, checkpoint, timeout_ms=1000)
    assert outcome.passed is False
    assert outcome.status == RunStatus.HARD_FAILURE
    assert outcome.code == OutcomeCode.CHECKPOINT_FAILED


async def test_probe_short_circuits_as_business_outcome_even_though_main_assertion_would_pass():
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("No member found", True)
    surface.script_checkpoint("/results", True)
    checkpoint = CheckpointSpec(
        description="reached results",
        assertion=UrlMatchesCheck(pattern="/results"),
        probes=[
            ConditionProbe(
                description="not-found message shown",
                assertion=TextPresentCheck(text="No member found"),
                status=RunStatus.BUSINESS_OUTCOME,
                code=OutcomeCode.CUSTOMER_NOT_FOUND,
            )
        ],
    )
    outcome = await CheckpointEngine().verify(surface, checkpoint, timeout_ms=1000)
    assert outcome.passed is False
    assert outcome.status == RunStatus.BUSINESS_OUTCOME
    assert outcome.code == OutcomeCode.CUSTOMER_NOT_FOUND


async def test_probe_miss_falls_through_to_main_assertion():
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("No member found", False)
    surface.script_checkpoint("/results", True)
    checkpoint = CheckpointSpec(
        description="reached results",
        assertion=UrlMatchesCheck(pattern="/results"),
        probes=[
            ConditionProbe(
                description="not-found message shown",
                assertion=TextPresentCheck(text="No member found"),
                status=RunStatus.BUSINESS_OUTCOME,
                code=OutcomeCode.CUSTOMER_NOT_FOUND,
            )
        ],
    )
    outcome = await CheckpointEngine().verify(surface, checkpoint, timeout_ms=1000)
    assert outcome.passed is True


def test_condition_probe_rejects_mismatched_status_and_code():
    with pytest.raises(ValidationError):
        ConditionProbe(
            description="bad",
            assertion=TextPresentCheck(text="x"),
            status=RunStatus.BUSINESS_OUTCOME,
            code=OutcomeCode.CHECKPOINT_FAILED,  # a HARD_FAILURE code
        )


def test_condition_probe_rejects_policy_blocked_status():
    with pytest.raises(ValidationError):
        ConditionProbe(
            description="bad",
            assertion=TextPresentCheck(text="x"),
            status=RunStatus.POLICY_BLOCKED,  # valid code for the status, but the status itself isn't a probe outcome
            code=OutcomeCode.DOMAIN_NOT_ALLOWED,
        )
