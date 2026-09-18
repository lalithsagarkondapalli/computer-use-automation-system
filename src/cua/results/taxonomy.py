"""Typed result contract and error taxonomy (Section 10).

This is the external contract that both discovery and replay return. The
assignment is explicit that debugging must not rely on exceptions alone,
and that a "business outcome" (a legitimate answer like customer-not-found)
must never be conflated with a crash. `RunStatus` is the coarse category a
caller branches on; `OutcomeCode` is the specific reason within that
category. The two are validated together so a `RunResult` can't claim,
say, `BUSINESS_OUTCOME` with a code that only makes sense as a
`HARD_FAILURE` -- the categories in the assignment (SUCCESS,
BUSINESS_OUTCOME, RECOVERABLE, POLICY_BLOCKED, ESCALATION_REQUIRED,
HARD_FAILURE) map exactly to the six `RunStatus` members below.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    BUSINESS_OUTCOME = "BUSINESS_OUTCOME"
    RECOVERABLE = "RECOVERABLE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    ESCALATION_REQUIRED = "ESCALATION_REQUIRED"
    HARD_FAILURE = "HARD_FAILURE"


class OutcomeCode(str, Enum):
    # BUSINESS_OUTCOME -- a legitimate, expected answer, not a failure.
    CUSTOMER_NOT_FOUND = "CUSTOMER_NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"

    # RECOVERABLE -- transient; a retry/dismiss strategy is expected to work.
    TRANSIENT_LOAD = "TRANSIENT_LOAD"
    KNOWN_INTERSTITIAL = "KNOWN_INTERSTITIAL"
    RETRYABLE_NAVIGATION = "RETRYABLE_NAVIGATION"

    # POLICY_BLOCKED -- the safety layer refused before anything executed.
    RISKY_ACTION_REQUIRES_APPROVAL = "RISKY_ACTION_REQUIRES_APPROVAL"
    DOMAIN_NOT_ALLOWED = "DOMAIN_NOT_ALLOWED"
    ACTION_TYPE_NOT_ALLOWED = "ACTION_TYPE_NOT_ALLOWED"

    # ESCALATION_REQUIRED -- automation cannot safely decide; needs a human.
    AMBIGUOUS_CONTROL = "AMBIGUOUS_CONTROL"
    UNKNOWN_DIALOG = "UNKNOWN_DIALOG"
    STUCK_STATE = "STUCK_STATE"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"
    # Added in Phase 4 -- discovery-specific stopping conditions that are
    # "automation cannot safely continue," not a crash, same rationale as
    # the four codes above (all pre-existing from Phase 2, so discovery's
    # need for them was anticipated before this phase's code existed).
    DISCOVERY_TIMEOUT = "DISCOVERY_TIMEOUT"  # overall wall-clock budget exceeded
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"  # bounded correction retries exhausted
    HUMAN_INTERVENTION_REQUESTED = "HUMAN_INTERVENTION_REQUESTED"  # model explicitly asked
    # Added in Phase 5: the human-in-the-loop handoff budget (repeated
    # resume attempts, or repeated distinct escalations) was exhausted --
    # automation cannot safely keep asking. See cua.handoff.controller.
    INTERVENTION_EXHAUSTED = "INTERVENTION_EXHAUSTED"

    # HARD_FAILURE -- stop and surface a clear, debuggable error.
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    SESSION_INVALID = "SESSION_INVALID"
    TARGET_UNRESOLVABLE = "TARGET_UNRESOLVABLE"
    INVALID_INPUT = "INVALID_INPUT"  # added in Phase 3: a caller-supplied invocation
    # parameter is missing or the wrong type. Distinct from INTERNAL_ERROR
    # (something broke inside) -- this is a contract violation by the
    # caller, caught before the surface is ever touched.
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Public (not the Phase 2 private `_VALID_CODES`): Phase 3's ConditionProbe
# (cua.artifact.checks) needs the same status/code mapping RunResult
# enforces, and duplicating the table would risk it drifting out of sync.
VALID_CODES_FOR_STATUS: dict[RunStatus, frozenset[OutcomeCode]] = {
    RunStatus.SUCCESS: frozenset(),
    RunStatus.BUSINESS_OUTCOME: frozenset(
        {OutcomeCode.CUSTOMER_NOT_FOUND, OutcomeCode.VALIDATION_ERROR, OutcomeCode.PERMISSION_DENIED}
    ),
    RunStatus.RECOVERABLE: frozenset(
        {OutcomeCode.TRANSIENT_LOAD, OutcomeCode.KNOWN_INTERSTITIAL, OutcomeCode.RETRYABLE_NAVIGATION}
    ),
    RunStatus.POLICY_BLOCKED: frozenset(
        {
            OutcomeCode.RISKY_ACTION_REQUIRES_APPROVAL,
            OutcomeCode.DOMAIN_NOT_ALLOWED,
            OutcomeCode.ACTION_TYPE_NOT_ALLOWED,
        }
    ),
    RunStatus.ESCALATION_REQUIRED: frozenset(
        {
            OutcomeCode.AMBIGUOUS_CONTROL,
            OutcomeCode.UNKNOWN_DIALOG,
            OutcomeCode.STUCK_STATE,
            OutcomeCode.MAX_STEPS_EXCEEDED,
            OutcomeCode.DISCOVERY_TIMEOUT,
            OutcomeCode.INVALID_MODEL_OUTPUT,
            OutcomeCode.HUMAN_INTERVENTION_REQUESTED,
            OutcomeCode.INTERVENTION_EXHAUSTED,
        }
    ),
    RunStatus.HARD_FAILURE: frozenset(
        {
            OutcomeCode.CHECKPOINT_FAILED,
            OutcomeCode.SESSION_INVALID,
            OutcomeCode.TARGET_UNRESOLVABLE,
            OutcomeCode.INVALID_INPUT,
            OutcomeCode.INTERNAL_ERROR,
        }
    ),
}


def code_is_valid_for_status(status: RunStatus, code: OutcomeCode) -> bool:
    return code in VALID_CODES_FOR_STATUS[status]


class EvidenceRef(BaseModel):
    """Pointer to a richer debugging artifact captured for a run (Section 3.5)."""

    kind: Literal["screenshot", "dom_snapshot", "trace", "log"]
    path: str


class RecoveryAttempt(BaseModel):
    """One attempted recovery action during a run (retry, dismiss, etc.)."""

    step_id: str
    strategy: str
    outcome: Literal["succeeded", "failed"]
    detail: str | None = None


class RunResult(BaseModel):
    """The single typed result contract returned by both discovery and replay.

    Every field a caller needs to safely branch on, debug, or audit a run
    lives here (Section 10) -- callers should never need to catch an
    exception to learn what happened.
    """

    run_id: str
    mode: Literal["discovery", "replay"]
    status: RunStatus
    code: OutcomeCode | None = None

    capability_id: str | None = None
    capability_version: str | None = None

    outputs: dict[str, Any] | None = None

    failing_step_id: str | None = None
    expected_state: str | None = None
    observed_state: str | None = None

    recovery_attempts: list[RecoveryAttempt] = Field(default_factory=list)
    escalation_ref: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)

    message: str = ""  # safe, redacted diagnostic -- never a raw exception string

    started_at: datetime
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _code_matches_status(self) -> "RunResult":
        if self.status == RunStatus.SUCCESS:
            if self.code is not None:
                raise ValueError("SUCCESS must not carry an outcome code")
        else:
            if self.code is None:
                raise ValueError(f"{self.status} requires an outcome code")
            if not code_is_valid_for_status(self.status, self.code):
                allowed = VALID_CODES_FOR_STATUS[self.status]
                raise ValueError(
                    f"code {self.code} is not valid for status {self.status}; "
                    f"allowed: {sorted(c.value for c in allowed)}"
                )
        return self
