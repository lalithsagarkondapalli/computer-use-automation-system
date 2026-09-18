"""Typed data the handoff mechanism passes around: the request that
starts an escalation, and the audit records kept of ownership changes and
human intervention (Sections 2 and 6 of the Phase 5 brief).

Deliberately independent of `cua.agent`/`cua.models`/`cua.replay` -- this
module (and the rest of `cua.handoff`) only ever imports
`cua.results.taxonomy` and `cua.safety.redaction`, so it is safe for both
`cua.agent` and `cua.replay` to depend on without either pulling in the
other's forbidden dependencies through it. See
tests/test_handoff_isolation.py for the automated proof.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from cua.handoff.state_machine import ControlState
from cua.results.taxonomy import EvidenceRef
from cua.safety.redaction import redact_text


class InterventionRequest(BaseModel):
    """Everything a human (or the operator console rendering this for a
    human) needs to understand why automation stopped and what it was
    trying to do -- and nothing more. Every free-text field is redacted at
    construction time (`create()`), not just when logged, since this
    object is also what the operator console renders directly."""

    intervention_id: str
    run_id: str
    mode: Literal["discovery", "replay"]
    capability_id: str | None = None
    capability_version: str | None = None
    goal: str | None = None  # discovery only
    step_id: str | None = None
    reason_code: str
    reason_message: str
    expected_state: str | None = None
    observed_state: str | None = None
    current_url: str
    evidence: EvidenceRef | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    control_owner: ControlState = ControlState.AUTOMATION_CONTROL

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        mode: Literal["discovery", "replay"],
        reason_code: str,
        reason_message: str,
        current_url: str,
        capability_id: str | None = None,
        capability_version: str | None = None,
        goal: str | None = None,
        step_id: str | None = None,
        expected_state: str | None = None,
        observed_state: str | None = None,
        evidence: EvidenceRef | None = None,
    ) -> "InterventionRequest":
        return cls(
            intervention_id=str(uuid.uuid4()),
            run_id=run_id,
            mode=mode,
            capability_id=capability_id,
            capability_version=capability_version,
            goal=redact_text(goal) if goal else goal,
            step_id=step_id,
            reason_code=reason_code,
            reason_message=redact_text(reason_message),
            expected_state=redact_text(expected_state) if expected_state else expected_state,
            observed_state=redact_text(observed_state) if observed_state else observed_state,
            current_url=current_url,
            evidence=evidence,
        )


class OwnershipTransition(BaseModel):
    """One entry in the audit trail of control changing hands."""

    from_state: ControlState
    to_state: ControlState
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InterventionRecord(BaseModel):
    """The audit record for one escalation, from takeover through
    resolution (Section 6). Deliberately does NOT attempt to log
    individual manual browser actions (clicks/keystrokes) -- see
    REPORT.md / the Phase 5 review notes for why that would require
    invasive instrumentation this system does not add. What is recorded
    is the aggregate: when control changed hands, what automation observed
    before and after, and how many resume attempts it took."""

    request: InterventionRequest
    takeover_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resume_requested_at: datetime | None = None
    resume_completed_at: datetime | None = None
    resume_attempts: int = 0
    resolved: bool | None = None  # None while still open
    state_before_url: str | None = None
    state_after_url: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
