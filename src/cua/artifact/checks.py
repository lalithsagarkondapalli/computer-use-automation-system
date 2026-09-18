"""Checkpoint assertions: typed postconditions a replay step can verify.

Kept as a closed, discriminated union rather than an arbitrary expression
language on purpose -- an artifact must not be able to smuggle arbitrary
code into a check (Section 6's "no arbitrary code execution from model
responses" applies just as much to what gets persisted as to what gets
proposed live), and a closed set is something both a human reviewer and
the ReplayEngine can reason about exhaustively. "Checkpoint" in the
glossary sense: a condition you assert to confirm you actually reached the
expected state, rather than assuming a click worked.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

from cua.artifact.targets import TargetDescriptor
from cua.results.taxonomy import OutcomeCode, RunStatus, code_is_valid_for_status


class UrlMatchesCheck(BaseModel):
    type: Literal["url_matches"] = "url_matches"
    pattern: str  # substring match against the current URL


class ElementVisibleCheck(BaseModel):
    type: Literal["element_visible"] = "element_visible"
    target: TargetDescriptor


class ElementNotVisibleCheck(BaseModel):
    type: Literal["element_not_visible"] = "element_not_visible"
    target: TargetDescriptor


class TextPresentCheck(BaseModel):
    type: Literal["text_present"] = "text_present"
    text: str


CheckpointAssertion = Annotated[
    Union[UrlMatchesCheck, ElementVisibleCheck, ElementNotVisibleCheck, TextPresentCheck],
    Field(discriminator="type"),
]


class ConditionProbe(BaseModel):
    """A named exceptional condition to watch for while a checkpoint waits.

    Added in Phase 3: a bare pass/fail checkpoint can't express "if you see
    this 'no member found' message, that's CUSTOMER_NOT_FOUND, not a
    failure" -- exactly the "business outcome vs. failure" distinction the
    assignment's glossary calls the most common design mistake in this
    space. `CheckpointEngine` (cua.replay.checkpoints) checks each probe,
    in order, before waiting on the checkpoint's own assertion; the first
    one observed short-circuits with this status/code instead.
    """

    description: str
    assertion: CheckpointAssertion
    status: RunStatus
    code: OutcomeCode
    probe_timeout_ms: int = Field(default=1500, ge=0, le=30_000)

    @model_validator(mode="after")
    def _status_and_code_are_a_valid_probe_outcome(self) -> "ConditionProbe":
        if self.status not in (RunStatus.BUSINESS_OUTCOME, RunStatus.ESCALATION_REQUIRED, RunStatus.RECOVERABLE):
            raise ValueError(
                f"a condition probe may only signal BUSINESS_OUTCOME, "
                f"ESCALATION_REQUIRED, or RECOVERABLE, not {self.status}"
            )
        if not code_is_valid_for_status(self.status, self.code):
            raise ValueError(f"{self.code} is not a valid code for status {self.status}")
        return self


class CheckpointSpec(BaseModel):
    description: str
    assertion: CheckpointAssertion
    probes: list[ConditionProbe] = Field(default_factory=list)
