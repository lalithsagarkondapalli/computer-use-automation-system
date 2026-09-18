"""ControlState: the explicit ownership state machine Section 3.3/Phase 5
requires -- automation and a human must never be able to both act on the
live session at once, and that guarantee has to be a small, checkable
table, not scattered `if` statements.

    AUTOMATION_CONTROL -> PAUSED_FOR_HUMAN -> HUMAN_CONTROL
    HUMAN_CONTROL -> RESUME_REQUESTED
    RESUME_REQUESTED -> AUTOMATION_CONTROL   (resume validated)
    RESUME_REQUESTED -> PAUSED_FOR_HUMAN     (resume rejected -- state still
                                               unresolved; bounces back to
                                               HUMAN_CONTROL for another
                                               attempt, bounded by
                                               HandoffController)

Every other transition (e.g. AUTOMATION_CONTROL -> HUMAN_CONTROL directly,
skipping the pause; or AUTOMATION_CONTROL -> RESUME_REQUESTED; or moving
"backwards") is rejected. This is the one piece of Phase 5 a reviewer can
verify by reading a five-line table rather than tracing control flow.
"""
from __future__ import annotations

from enum import Enum


class ControlState(str, Enum):
    AUTOMATION_CONTROL = "AUTOMATION_CONTROL"
    PAUSED_FOR_HUMAN = "PAUSED_FOR_HUMAN"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    RESUME_REQUESTED = "RESUME_REQUESTED"


VALID_TRANSITIONS: dict[ControlState, frozenset[ControlState]] = {
    ControlState.AUTOMATION_CONTROL: frozenset({ControlState.PAUSED_FOR_HUMAN}),
    ControlState.PAUSED_FOR_HUMAN: frozenset({ControlState.HUMAN_CONTROL}),
    ControlState.HUMAN_CONTROL: frozenset({ControlState.RESUME_REQUESTED}),
    ControlState.RESUME_REQUESTED: frozenset({ControlState.AUTOMATION_CONTROL, ControlState.PAUSED_FOR_HUMAN}),
}


class InvalidTransition(Exception):
    def __init__(self, current: ControlState, attempted: ControlState):
        super().__init__(f"cannot transition {current.value} -> {attempted.value}")
        self.current = current
        self.attempted = attempted


def is_valid_transition(current: ControlState, attempted: ControlState) -> bool:
    return attempted in VALID_TRANSITIONS.get(current, frozenset())
