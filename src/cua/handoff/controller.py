"""HandoffController: the runtime that owns the control-ownership state
machine, blocks automation while a human has the session, and validates a
resume before ever handing control back (Sections 3.3/3.5, Phase 5).

Session continuity is guaranteed by what this module does NOT do: it never
constructs, launches, or holds a `SurfaceAdapter`, `Page`, or
`BrowserContext` of its own (see tests/test_handoff_isolation.py, which
asserts `cua.handoff` never imports `cua.surfaces.playwright_web` -- there
is structurally no code path here that could open a second browser). The
one live session -- the same `SurfaceAdapter` object -- stays entirely in
the hands of whoever is calling `escalate_and_wait` (`DiscoveryLoop` or
`ReplayEngine`), and this controller only ever gates *when* that caller is
allowed to keep acting on it and *whether* a proposed resume is optional
(e.g. discovery already has a model that will re-assess next) or a
positional recheck this controller must itself confirm (e.g. deterministic
replay, which has no model to fall back on).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from cua.handoff.models import InterventionRecord, InterventionRequest, OwnershipTransition
from cua.handoff.state_machine import ControlState, InvalidTransition, is_valid_transition

ResumeValidator = Callable[[], Awaitable[bool]]


class ControlViolation(Exception):
    """Raised when automation attempts (or would attempt) a live-surface
    action while it does not hold AUTOMATION_CONTROL. This is the
    "impossible by construction" half of Section 5's pause requirement --
    `wait_until_automation_control()` is the "actually blocks" half."""


class InterventionBudgetExhausted(Exception):
    """Raised by `escalate_and_wait` when the total number of distinct
    escalations for this controller's run has already reached its bound
    (Section 7: "prevent infinite human-resume loops")."""


class HandoffController:
    def __init__(self, *, max_interventions: int = 3, max_resume_attempts: int = 3):
        self._state = ControlState.AUTOMATION_CONTROL
        self._transitions: list[OwnershipTransition] = []
        self._interventions: list[InterventionRecord] = []
        self._max_interventions = max_interventions
        self._max_resume_attempts = max_resume_attempts
        self._automation_gate = asyncio.Event()
        self._automation_gate.set()
        self._resume_signal = asyncio.Event()

    @property
    def state(self) -> ControlState:
        return self._state

    @property
    def transitions(self) -> list[OwnershipTransition]:
        return list(self._transitions)

    @property
    def interventions(self) -> list[InterventionRecord]:
        return list(self._interventions)

    @property
    def current_intervention(self) -> InterventionRecord | None:
        return self._interventions[-1] if self._interventions else None

    def assert_automation_control(self) -> None:
        """Called immediately before any live-surface action. Raises
        unconditionally if automation does not currently hold control --
        this is what makes "an automated action attempt should be
        rejected or impossible by construction" true even if a caller's
        own gating logic has a bug, not merely documented."""
        if self._state != ControlState.AUTOMATION_CONTROL:
            raise ControlViolation(f"automation attempted a live-surface action while state={self._state.value}")

    async def wait_until_automation_control(self) -> None:
        """Blocks via a real asyncio.Event, not a sleep/poll loop, until
        control is back with automation."""
        await self._automation_gate.wait()

    def _transition(self, to: ControlState) -> None:
        if not is_valid_transition(self._state, to):
            raise InvalidTransition(self._state, to)
        self._transitions.append(OwnershipTransition(from_state=self._state, to_state=to))
        self._state = to

    async def escalate(self, request: InterventionRequest) -> None:
        """AUTOMATION_CONTROL -> PAUSED_FOR_HUMAN -> HUMAN_CONTROL, as two
        distinct, separately-logged transitions -- "detects escalation,
        pauses, [then] explicitly cedes control" (Section: core
        requirement) is three described steps, and the pause and cede are
        the two this method is responsible for; "detects" already
        happened in the caller."""
        if len(self._interventions) >= self._max_interventions:
            raise InterventionBudgetExhausted(
                f"this run has already escalated {len(self._interventions)} times "
                f"(max_interventions={self._max_interventions})"
            )
        self._automation_gate.clear()
        self._transition(ControlState.PAUSED_FOR_HUMAN)
        self._transition(ControlState.HUMAN_CONTROL)
        self._resume_signal.clear()
        record = InterventionRecord(request=request, state_before_url=request.current_url)
        if request.evidence is not None:
            record.evidence.append(request.evidence)
        self._interventions.append(record)

    def request_resume(self) -> None:
        """The human (or the operator console, on their behalf) signals
        they are done. HUMAN_CONTROL -> RESUME_REQUESTED. Synchronous and
        side-effect-free beyond the state transition and signaling the
        internal event -- safe to call from the console's own thread
        (see cua.handoff.console), since asyncio.Event.set() itself is not
        thread-safe, so the console schedules this through the owning
        event loop rather than calling it directly from another thread."""
        self._transition(ControlState.RESUME_REQUESTED)
        record = self.current_intervention
        if record is not None:
            record.resume_requested_at = datetime.now(timezone.utc)
            record.resume_attempts += 1
        self._resume_signal.set()

    async def await_resume(
        self, validate: ResumeValidator | None, *, current_url_fn: Callable[[], Awaitable[str]] | None = None
    ) -> bool:
        """Waits for `request_resume()`, then applies `validate` (if any)
        before actually granting control back -- clicking Resume must not
        blindly mean "fixed" (Section 7). `validate=None` is a deliberate,
        documented choice for discovery: the model itself re-observes and
        re-decides next, so there is no fixed condition for the controller
        to check here. Replay always passes a real validator, since replay
        has no fallback judgment of its own.

        `current_url_fn`, if given, is called AFTER the resume signal
        fires -- not before -- so the recorded `state_after_url` reflects
        what the surface looks like once the human is actually done, not
        a stale snapshot taken while this call was still waiting. (An
        earlier version called it before waiting; a real-browser test
        caught the resulting stale URL -- see tests/test_handoff_playwright_integration.py
        and the Phase 5 review notes' "Bugs discovered" section.)

        On success: RESUME_REQUESTED -> AUTOMATION_CONTROL, gate opens.
        On failure: RESUME_REQUESTED -> PAUSED_FOR_HUMAN -> HUMAN_CONTROL
        (bounced back for another attempt) and returns False -- the caller
        (`escalate_and_wait`) is what bounds how many times this can
        happen.
        """
        await self._resume_signal.wait()
        current_url = await current_url_fn() if current_url_fn is not None else None
        ok = True if validate is None else await validate()
        record = self.current_intervention
        if record is not None:
            record.resolved = ok
            record.resume_completed_at = datetime.now(timezone.utc)
            if current_url is not None:
                record.state_after_url = current_url
        if ok:
            self._transition(ControlState.AUTOMATION_CONTROL)
            self._automation_gate.set()
        else:
            self._transition(ControlState.PAUSED_FOR_HUMAN)
            self._transition(ControlState.HUMAN_CONTROL)
            self._resume_signal.clear()
        return ok

    async def escalate_and_wait(
        self, request: InterventionRequest, validate: ResumeValidator | None = None, *, current_url_fn: Callable[[], Awaitable[str]] | None = None
    ) -> bool:
        """The convenience most callers use: escalate once, then wait for
        up to `max_resume_attempts` human resume signals, validating each
        one, until either a validated resume succeeds or the bound is
        exhausted (never an unbounded loop -- Section 7)."""
        await self.escalate(request)
        for _ in range(self._max_resume_attempts):
            ok = await self.await_resume(validate, current_url_fn=current_url_fn)
            if ok:
                return True
        return False
