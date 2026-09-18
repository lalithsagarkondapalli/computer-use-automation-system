"""The shared, closed action vocabulary.

Both a persisted `CapabilityArtifact` step (see `cua.artifact.schema.Step`)
and a live discovery-agent proposal (see `cua.agent.actions.AgentAction`)
are expressed in terms of this same enum -- it is the single constrained
vocabulary the assignment (Section 6) asks for, so the model can only ever
propose one of these nine things, never arbitrary code or free-form text.

Two members, FINISH and REQUEST_HUMAN, are agent-loop *control* actions:
they end or escalate a discovery run and are meaningless as a replay step
(there is nothing to replay about "the goal is done" or "ask a human").
`STEP_ACTIONS` is the subset valid inside a persisted artifact; this is
enforced by `Step`'s validator, not just documented here.
"""
from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    READ_EXTRACT = "read_extract"
    WAIT = "wait"
    CHECKPOINT = "checkpoint"
    FINISH = "finish"
    REQUEST_HUMAN = "request_human"


STEP_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.NAVIGATE,
        ActionType.CLICK,
        ActionType.FILL,
        ActionType.SELECT,
        ActionType.READ_EXTRACT,
        ActionType.WAIT,
        ActionType.CHECKPOINT,
    }
)
