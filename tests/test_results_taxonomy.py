from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cua.results.taxonomy import OutcomeCode, RunResult, RunStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_success_without_code_is_valid():
    RunResult(run_id="r1", mode="replay", status=RunStatus.SUCCESS, started_at=_now())


def test_success_with_code_rejected():
    with pytest.raises(ValidationError):
        RunResult(
            run_id="r1",
            mode="replay",
            status=RunStatus.SUCCESS,
            code=OutcomeCode.CUSTOMER_NOT_FOUND,
            started_at=_now(),
        )


def test_business_outcome_requires_a_code():
    with pytest.raises(ValidationError):
        RunResult(run_id="r1", mode="replay", status=RunStatus.BUSINESS_OUTCOME, started_at=_now())


def test_business_outcome_with_matching_code_is_valid():
    RunResult(
        run_id="r1",
        mode="replay",
        status=RunStatus.BUSINESS_OUTCOME,
        code=OutcomeCode.CUSTOMER_NOT_FOUND,
        started_at=_now(),
    )


def test_code_from_wrong_category_rejected():
    with pytest.raises(ValidationError):
        RunResult(
            run_id="r1",
            mode="replay",
            status=RunStatus.BUSINESS_OUTCOME,
            code=OutcomeCode.CHECKPOINT_FAILED,  # a HARD_FAILURE code, not a business outcome
            started_at=_now(),
        )


def test_escalation_required_with_ambiguous_control():
    result = RunResult(
        run_id="r1",
        mode="replay",
        status=RunStatus.ESCALATION_REQUIRED,
        code=OutcomeCode.AMBIGUOUS_CONTROL,
        started_at=_now(),
        escalation_ref="intervention-42",
    )
    assert result.escalation_ref == "intervention-42"
