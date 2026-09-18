"""The discovery transcript: what happened at each step of a discovery
run, structured enough that `ArtifactBuilder` (Section 6) can turn a
successful run into a `CapabilityArtifact` without re-deriving anything
from prose, and complete enough that a human reviewer can audit *why*
discovery did what it did without replaying the raw model conversation.

Deliberately does NOT store hidden/internal model reasoning (Section 3.5:
"do not persist full chain-of-thought") -- `AgentAction.reasoning` is
already a short, required, action-oriented field (see its docstring), and
that is the only "why" this trace carries.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from cua.agent.actions import AgentAction
from cua.artifact.checks import CheckpointAssertion
from cua.results.taxonomy import EvidenceRef
from cua.safety.models import PolicyDecision
from cua.surfaces.observation import Observation


class DiscoveryStopReason(str, Enum):
    GOAL_ACHIEVED = "GOAL_ACHIEVED"
    MAX_STEPS = "MAX_STEPS"
    TIMEOUT = "TIMEOUT"
    STUCK_STATE = "STUCK_STATE"
    INVALID_OUTPUT_EXHAUSTED = "INVALID_OUTPUT_EXHAUSTED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    SURFACE_ERROR = "SURFACE_ERROR"
    HUMAN_REQUESTED = "HUMAN_REQUESTED"
    # Phase 5: HUMAN_REQUESTED and STUCK_STATE are handoff-eligible (see
    # cua.agent.loop.DiscoveryLoop._HANDOFF_ELIGIBLE); this is the stop
    # reason used only when a handoff was attempted and its resume budget
    # was exhausted rather than validated -- automation cannot safely keep
    # asking. Mirrors OutcomeCode.INTERVENTION_EXHAUSTED (cua.results.taxonomy),
    # which is replay's equivalent for the same underlying budget.
    INTERVENTION_EXHAUSTED = "INTERVENTION_EXHAUSTED"


class DiscoveryStepRecord(BaseModel):
    """One full OBSERVE -> DECIDE -> VALIDATE -> POLICY CHECK -> ACT cycle.

    Exactly one of (executed=True) or (validation_error / policy_decision
    with allowed=False / execution_error) is populated -- a step that
    didn't execute always says why, at the point it stopped."""

    step_number: int
    observation_before: Observation
    action: AgentAction | None = None  # None only if invalid-output retries were exhausted this step
    invalid_output_attempts: int = 0
    validation_error: str | None = None
    policy_decision: PolicyDecision | None = None
    executed: bool = False
    execution_error: str | None = None
    observation_after: Observation | None = None
    checkpoint_assertion: CheckpointAssertion | None = None  # set only for an executed CHECKPOINT
    extracted_output_name: str | None = None  # set only for an executed READ_EXTRACT
    extracted_value: str | None = None


class DiscoveryTrace(BaseModel):
    run_id: str
    goal: str
    steps: list[DiscoveryStepRecord] = Field(default_factory=list)
    stop_reason: DiscoveryStopReason | None = None
    stop_message: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)

    def successfully_executed_steps(self) -> list[DiscoveryStepRecord]:
        return [s for s in self.steps if s.executed]
