"""The real-browser same-session handoff test (Phase 5, Section 10's
explicit ask): a single real headless Chromium `PlaywrightWebSurface` is
opened ONCE. Automation (a real `ReplayEngine`) reaches a genuinely
blocked state and pauses; test code -- standing in for a human -- then
acts directly on the SAME `SurfaceAdapter` object's existing public
methods (`resolve` / `click`), using no new abstract methods and no
second browser/page/context; resume is requested; automation re-observes
the SAME live page and continues to a real SUCCESS.

The blocking condition is not staged or faked: `sample_page.html`'s
loading spinner really is visible until something clicks "Search," and
`CheckpointEngine` really does check the escalation-flagged
`ElementVisibleCheck(Loading)` probe before its `UrlMatchesCheck("#results")`
assertion. Nothing about the escalation trigger here is scripted separately
from what the real DOM is actually doing at that moment.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from cua.actions import ActionType
from cua.artifact.checks import CheckpointSpec, ConditionProbe, ElementVisibleCheck, UrlMatchesCheck
from cua.artifact.schema import CapabilityArtifact, Step
from cua.artifact.targets import TargetDescriptor
from cua.handoff import ControlState, HandoffController
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.surfaces.playwright_web import PlaywrightWebSurface

from factories import base_kwargs

FIXTURE_PATH = (Path(__file__).parent / "fixtures" / "sample_page.html").resolve()
_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")


def _blocked_checkpoint_artifact() -> CapabilityArtifact:
    """One step: a checkpoint that cannot pass until something clicks
    "Search" -- and whose own ConditionProbe (the still-visible loading
    spinner) is what turns that "not yet" into an ESCALATION_REQUIRED
    rather than a plain timeout. Deliberately has no NAVIGATE/CLICK step
    of its own -- reaching #results is left entirely to the human."""
    steps = [
        Step(
            step_id="cp_results",
            action=ActionType.CHECKPOINT,
            checkpoint=CheckpointSpec(
                description="results page reached (#results in the URL)",
                assertion=UrlMatchesCheck(pattern="#results"),
                probes=[
                    ConditionProbe(
                        description="loading spinner never cleared -- needs a human to look",
                        assertion=ElementVisibleCheck(target=TargetDescriptor(role="status", accessible_name="Loading")),
                        status=RunStatus.ESCALATION_REQUIRED,
                        code=OutcomeCode.UNKNOWN_DIALOG,
                    )
                ],
                probe_timeout_ms=500,
            ),
            timeout_ms=800,
        ),
    ]
    kwargs = base_kwargs(steps)
    kwargs["inputs"] = []
    return CapabilityArtifact(**kwargs)


def _make_engine(tmp_path: Path, handoff: HandoffController) -> ReplayEngine:
    policy = PolicyEngine(PolicyConfig(
        allowed_domains=("127.0.0.1",),  # matches the artifact's declared target -- the live page is file://, never checked against this
        allowed_actions=(ActionType.CHECKPOINT, ActionType.CLICK),
    ))
    logger = StructuredLogger(run_id="playwright-handoff-test", log_dir=tmp_path / "logs")
    evidence = EvidenceCapture(tmp_path / "evidence")
    return ReplayEngine(policy, logger, evidence, handoff)


async def test_real_browser_same_session_handoff_escalate_pause_human_acts_resume_succeed(tmp_path: Path):
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    try:
        await surface.navigate(f"file://{FIXTURE_PATH}")
        # sanity: the spinner really is visible and #results is really not
        # reached yet, before any escalation logic runs at all
        assert "#results" not in (await surface.current_url())

        handoff = HandoffController()
        engine = _make_engine(tmp_path, handoff)

        run_task = asyncio.ensure_future(engine.run(_blocked_checkpoint_artifact(), {}, surface))

        # wait for automation to genuinely pause -- not a fixed sleep, a
        # real poll on the real controller state (Section 5: this is a
        # readiness wait for the test, not the pause mechanism itself,
        # which is the controller's own asyncio.Event)
        for _ in range(200):
            if handoff.state == ControlState.HUMAN_CONTROL:
                break
            await asyncio.sleep(0.02)
        assert handoff.state == ControlState.HUMAN_CONTROL
        assert not run_task.done()  # automation is genuinely blocked, not racing ahead

        # -- the "human" now acts directly on the SAME live Page/session,
        # through the SurfaceAdapter's own existing public methods -- no
        # second browser, no reload, no new abstract method --
        intervention = handoff.current_intervention
        assert intervention is not None
        assert intervention.request.reason_code == OutcomeCode.UNKNOWN_DIALOG.value
        assert "file://" in intervention.request.current_url  # proves it's this same page, not a placeholder

        resolved = await surface.resolve(TargetDescriptor(role="button", accessible_name="Search"))
        await surface.click(resolved)
        # the human's click is a REAL DOM mutation on the REAL page --
        # confirm it before ever touching the controller, so the
        # assertions below about "automation observed the human's change"
        # aren't circular
        assert "#results" in (await surface.current_url())

        handoff.request_resume()
        result = await asyncio.wait_for(run_task, timeout=10)

        assert result.status == RunStatus.SUCCESS
        assert handoff.state == ControlState.AUTOMATION_CONTROL
        assert len(handoff.interventions) == 1
        record = handoff.interventions[0]
        assert record.resolved is True
        assert record.state_before_url is not None and "#results" not in record.state_before_url
        assert record.state_after_url is not None and "#results" in record.state_after_url
    finally:
        await surface.close()


async def test_real_browser_handoff_without_the_human_acting_does_not_falsely_succeed(tmp_path: Path):
    """The negative control for the test above: if the human clicks
    Resume without actually doing anything (the spinner is still up,
    #results was never reached), the deterministic recheck must catch
    that and NOT report success -- proving the first test's SUCCESS came
    from the human's real action, not from resume alone."""
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    try:
        await surface.navigate(f"file://{FIXTURE_PATH}")
        handoff = HandoffController(max_resume_attempts=1)
        engine = _make_engine(tmp_path, handoff)

        run_task = asyncio.ensure_future(engine.run(_blocked_checkpoint_artifact(), {}, surface))
        for _ in range(200):
            if handoff.state == ControlState.HUMAN_CONTROL:
                break
            await asyncio.sleep(0.02)
        assert handoff.state == ControlState.HUMAN_CONTROL

        handoff.request_resume()  # the human signals "done" without acting
        result = await asyncio.wait_for(run_task, timeout=10)

        assert result.status == RunStatus.ESCALATION_REQUIRED
        assert result.code == OutcomeCode.INTERVENTION_EXHAUSTED
    finally:
        await surface.close()
