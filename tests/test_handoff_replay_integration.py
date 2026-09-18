"""Replay <-> handoff integration (Phase 5, Section 9): a real
`HandoffController` wired into `ReplayEngine`, covering both escalation
paths -- an ambiguous target (`TargetAmbiguous`) and a checkpoint's own
`ConditionProbe`-triggered ESCALATION_REQUIRED outcome -- with ZERO model
decisions anywhere in the path. Every resume validator here is a plain,
deterministic recheck against the same FakeSurfaceAdapter: re-`resolve()`
the target, or re-`verify()` the checkpoint. tests/test_replay_model_isolation.py
and tests/test_handoff_isolation.py are the standing, automated proof that
this integration never imports `cua.agent`/`cua.models`; this file proves
the runtime *behavior* on top of that static guarantee.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from cua.actions import ActionType
from cua.artifact.checks import CheckpointSpec, ConditionProbe, TextPresentCheck, UrlMatchesCheck
from cua.artifact.schema import CapabilityArtifact, Step
from cua.artifact.targets import TargetDescriptor
from cua.handoff import ControlState, HandoffController
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig

from factories import base_kwargs
from fakes import FakeElement, FakeSurfaceAdapter


def _make_engine(tmp_path: Path, handoff: HandoffController | None = None, **policy_overrides) -> ReplayEngine:
    defaults = dict(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                          ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT),
    )
    defaults.update(policy_overrides)
    policy = PolicyEngine(PolicyConfig(**defaults))
    logger = StructuredLogger(run_id="replay-handoff-test", log_dir=tmp_path / "logs")
    evidence = EvidenceCapture(tmp_path / "evidence")
    return ReplayEngine(policy, logger, evidence, handoff)


def _minimal_artifact(steps: list[Step]) -> CapabilityArtifact:
    kwargs = base_kwargs(steps)
    kwargs["inputs"] = []
    return CapabilityArtifact(**kwargs)


def _ambiguous_click_artifact() -> CapabilityArtifact:
    steps = [
        Step(step_id="click_search", action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search")),
        Step(step_id="cp", action=ActionType.CHECKPOINT, checkpoint=CheckpointSpec(description="done", assertion=UrlMatchesCheck(pattern="/x"))),
    ]
    return _minimal_artifact(steps)


async def test_ambiguous_target_resolved_by_a_human_lets_replay_continue_deterministically(tmp_path: Path):
    """The human doesn't fix anything through this codebase -- the test
    itself plays the human by mutating what the SAME FakeSurfaceAdapter
    resolves, exactly the way a person disambiguating a real page would
    change what's on it. The validator that decides whether to resume is
    a plain re-`resolve()` call, not a model."""
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])  # ambiguous: 2 candidates
    surface.script_checkpoint("/x", True)
    handoff = HandoffController()
    engine = _make_engine(tmp_path, handoff=handoff)

    async def human_disambiguates_then_resumes() -> None:
        await asyncio.sleep(0)
        assert handoff.state == ControlState.HUMAN_CONTROL
        surface.register_role_name("button", "Search", [FakeElement()])  # now exactly one candidate
        handoff.request_resume()

    result, _ = await asyncio.gather(
        engine.run(_ambiguous_click_artifact(), {}, surface),
        human_disambiguates_then_resumes(),
    )

    assert result.status == RunStatus.SUCCESS
    assert len(handoff.interventions) == 1
    assert handoff.interventions[0].resolved is True
    assert handoff.interventions[0].request.reason_code == OutcomeCode.AMBIGUOUS_CONTROL.value
    assert handoff.state == ControlState.AUTOMATION_CONTROL


async def test_ambiguous_target_that_stays_ambiguous_after_resume_fails_not_hangs(tmp_path: Path):
    """Clicking Resume must not blindly mean fixed (Section 7): if the
    human signals done but the ambiguity is still there, the resume
    validator's own deterministic recheck (`surface.resolve` still raising
    TargetAmbiguous) reports it unresolved; once the bounded resume-attempt
    budget for this call is spent, replay reports a clean, typed failure --
    it must not loop forever or silently succeed."""
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])  # stays ambiguous
    handoff = HandoffController(max_resume_attempts=1)
    engine = _make_engine(tmp_path, handoff=handoff)

    async def human_resumes_without_fixing_anything() -> None:
        await asyncio.sleep(0)
        handoff.request_resume()

    result, _ = await asyncio.gather(
        engine.run(_ambiguous_click_artifact(), {}, surface),
        human_resumes_without_fixing_anything(),
    )

    assert result.status == RunStatus.ESCALATION_REQUIRED
    # the handoff's own resume-attempt budget (not a second, separate
    # deterministic retry) is what was exhausted here -- the validator
    # already reconfirmed the ambiguity before ever returning control
    assert result.code == OutcomeCode.INTERVENTION_EXHAUSTED
    assert len(handoff.interventions) == 1
    assert handoff.interventions[0].resolved is False


async def test_ambiguous_target_without_a_handoff_behaves_exactly_as_phase3(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])
    engine = _make_engine(tmp_path, handoff=None)

    result = await engine.run(_ambiguous_click_artifact(), {}, surface)

    assert result.status == RunStatus.ESCALATION_REQUIRED
    assert result.code == OutcomeCode.AMBIGUOUS_CONTROL


async def test_checkpoint_probe_escalation_resolved_by_a_human_lets_replay_continue(tmp_path: Path):
    """The other Section 9 path: a ConditionProbe classifies the observed
    state as ESCALATION_REQUIRED (e.g. an unrecognized dialog). The resume
    validator deterministically re-verifies the SAME checkpoint."""
    steps = [
        Step(
            step_id="cp", action=ActionType.CHECKPOINT,
            checkpoint=CheckpointSpec(
                description="results reached",
                assertion=UrlMatchesCheck(pattern="/results"),
                probes=[ConditionProbe(
                    description="unrecognized dialog blocking the page",
                    assertion=TextPresentCheck(text="Unexpected dialog"),
                    status=RunStatus.ESCALATION_REQUIRED,
                    code=OutcomeCode.UNKNOWN_DIALOG,
                )],
            ),
        ),
    ]
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("Unexpected dialog", True)  # probe fires first
    surface.script_checkpoint("/results", False)  # main assertion not yet true
    handoff = HandoffController()
    engine = _make_engine(tmp_path, handoff=handoff)

    async def human_dismisses_dialog_then_resumes() -> None:
        await asyncio.sleep(0)
        assert handoff.state == ControlState.HUMAN_CONTROL
        surface.script_checkpoint("Unexpected dialog", False)  # human dismissed it
        surface.script_checkpoint("/results", True)  # now the real condition holds
        handoff.request_resume()

    result, _ = await asyncio.gather(
        engine.run(_minimal_artifact(steps), {}, surface),
        human_dismisses_dialog_then_resumes(),
    )

    assert result.status == RunStatus.SUCCESS
    assert len(handoff.interventions) == 1
    assert handoff.interventions[0].request.reason_code == OutcomeCode.UNKNOWN_DIALOG.value


async def test_checkpoint_probe_escalation_exhausted_becomes_intervention_exhausted(tmp_path: Path):
    steps = [
        Step(
            step_id="cp", action=ActionType.CHECKPOINT,
            checkpoint=CheckpointSpec(
                description="results reached",
                assertion=UrlMatchesCheck(pattern="/results"),
                probes=[ConditionProbe(
                    description="unrecognized dialog blocking the page",
                    assertion=TextPresentCheck(text="Unexpected dialog"),
                    status=RunStatus.ESCALATION_REQUIRED,
                    code=OutcomeCode.UNKNOWN_DIALOG,
                )],
            ),
        ),
    ]
    surface = FakeSurfaceAdapter()
    surface.script_checkpoint("Unexpected dialog", True)  # never cleared
    handoff = HandoffController(max_resume_attempts=1)
    engine = _make_engine(tmp_path, handoff=handoff)

    async def human_resumes_without_fixing_anything() -> None:
        await asyncio.sleep(0)
        handoff.request_resume()

    result, _ = await asyncio.gather(
        engine.run(_minimal_artifact(steps), {}, surface),
        human_resumes_without_fixing_anything(),
    )

    assert result.status == RunStatus.ESCALATION_REQUIRED
    assert result.code == OutcomeCode.INTERVENTION_EXHAUSTED


async def test_same_surface_adapter_object_survives_the_entire_handoff(tmp_path: Path):
    """Section 3, made concrete for replay: the exact same FakeSurfaceAdapter
    instance is what both the engine and the "human" (test code standing in
    for one) touch -- there is no second surface anywhere in this path."""
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])
    surface.script_checkpoint("/x", True)
    handoff = HandoffController()
    engine = _make_engine(tmp_path, handoff=handoff)
    seen_surface_ids: list[int] = []

    async def human_disambiguates_then_resumes() -> None:
        await asyncio.sleep(0)
        seen_surface_ids.append(id(surface))
        surface.register_role_name("button", "Search", [FakeElement()])
        handoff.request_resume()

    result, _ = await asyncio.gather(
        engine.run(_ambiguous_click_artifact(), {}, surface),
        human_disambiguates_then_resumes(),
    )

    assert result.status == RunStatus.SUCCESS
    assert seen_surface_ids == [id(surface)]
    # the engine's own retry after resume necessarily used this exact
    # object too, since it's the only surface argument `run()` was ever given
