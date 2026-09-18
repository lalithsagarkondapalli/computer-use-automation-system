"""Human-in-the-loop control-transfer mechanism (Phase 5).

State machine (`cua.handoff.state_machine.ControlState`):
AUTOMATION_CONTROL -> PAUSED_FOR_HUMAN -> HUMAN_CONTROL -> RESUME_REQUESTED
-> AUTOMATION_CONTROL (or back to PAUSED_FOR_HUMAN if the resume doesn't
validate), enforced by `cua.handoff.controller.HandoffController`, plus a
minimal local operator console (`cua.handoff.console.OperatorConsole`,
stdlib HTTP only -- no FastAPI/uvicorn, since nothing here needed them)
that lets a human see why automation stopped and click Resume, while
operating the already-visible live browser session directly.

Deliberately independent of `cua.agent`, `cua.models`, and `cua.replay`
(see tests/test_handoff_isolation.py) so both the discovery loop and the
replay engine can depend on this package without either pulling in the
other's forbidden dependencies through it.
"""
from cua.handoff.controller import ControlViolation, HandoffController, InterventionBudgetExhausted
from cua.handoff.models import InterventionRecord, InterventionRequest, OwnershipTransition
from cua.handoff.state_machine import ControlState, InvalidTransition

__all__ = [
    "ControlState",
    "ControlViolation",
    "HandoffController",
    "InterventionBudgetExhausted",
    "InterventionRecord",
    "InterventionRequest",
    "InvalidTransition",
    "OwnershipTransition",
]
