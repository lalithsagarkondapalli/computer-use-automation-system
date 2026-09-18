"""Discovery <-> handoff integration (Phase 5, Section 8): only
HUMAN_REQUESTED and STUCK_STATE are handoff-eligible; a real
`HandoffController` is wired in (not mocked) and driven the same way
tests/test_handoff_controller.py drives it directly -- a concurrent
coroutine plays the human by calling `request_resume()` while
`DiscoveryLoop.run()` is genuinely blocked waiting for it via a real
`asyncio.Event`, not a poll loop and not a stub.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.agent.trace import DiscoveryStopReason
from cua.artifact.targets import TargetDescriptor
from cua.handoff import ControlState, HandoffController
from cua.observability.logger import StructuredLogger
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig

from discovery_fixtures import BASE
from discovery_fixtures import obs as _obs
from fake_model import FakeModelAdapter
from fakes import FakeSurfaceAdapter


def _make_loop(surface: FakeSurfaceAdapter, model: FakeModelAdapter, tmp_path: Path, **overrides) -> DiscoveryLoop:
    policy = PolicyEngine(
        PolicyConfig(
            allowed_domains=("127.0.0.1",),
            allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                              ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT),
        )
    )
    logger = StructuredLogger(run_id="discovery-handoff-test", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    defaults = dict(max_steps=10, overall_timeout_s=300, stuck_repeat_threshold=3, max_invalid_output_retries=2)
    defaults.update(overrides)
    return DiscoveryLoop(model, surface, policy, logger, evidence, **defaults)


async def test_human_requested_escalates_and_resumes_via_a_real_controller(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "unexpected compliance-hold dialog"}, reasoning="need help"),
        AgentAction(action=ActionType.FINISH, reasoning="done after the human cleared the hold"),
    ])
    handoff = HandoffController()
    loop = _make_loop(surface, model, tmp_path, handoff=handoff)

    async def operator_resumes() -> None:
        await asyncio.sleep(0)
        assert handoff.state == ControlState.HUMAN_CONTROL  # automation really is paused right now
        handoff.request_resume()

    trace, _ = await asyncio.gather(loop.run("look up a member"), operator_resumes())

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert handoff.state == ControlState.AUTOMATION_CONTROL
    assert len(handoff.interventions) == 1
    record = handoff.interventions[0]
    assert record.resolved is True
    assert record.request.reason_code == "HUMAN_REQUESTED"
    assert record.request.mode == "discovery"
    assert record.request.goal == "look up a member"
    # not automatically used for every failure -- REQUEST_HUMAN specifically is what triggered this


async def test_stuck_state_escalates_and_a_resumed_human_lets_discovery_continue(tmp_path: Path):
    from cua.surfaces.observation import InteractiveElement

    btn = InteractiveElement(role="button", accessible_name="Retry")
    surface = FakeSurfaceAdapter()
    for _ in range(4):
        surface.script_observation(_obs(f"{BASE}/", [btn]))
    surface.register_role_name("button", "Retry", [])  # never resolves -- always TargetNotFound
    action = AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Retry"), reasoning="try again")
    finish = AgentAction(action=ActionType.FINISH, reasoning="the human fixed it")
    model = FakeModelAdapter([action, action, action, finish])
    handoff = HandoffController()
    loop = _make_loop(surface, model, tmp_path, handoff=handoff, stuck_repeat_threshold=3)

    async def operator_resumes() -> None:
        await asyncio.sleep(0)
        handoff.request_resume()

    trace, _ = await asyncio.gather(loop.run("goal"), operator_resumes())

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert len(handoff.interventions) == 1
    assert handoff.interventions[0].request.reason_code == "STUCK_STATE"


async def test_automation_cannot_act_while_paused_for_the_human(tmp_path: Path):
    """The "impossible by construction" half of Section 5: while the
    controller is in HUMAN_CONTROL, DiscoveryLoop's own `_guard()` calls
    would raise if the loop somehow tried to touch the surface -- proven
    directly against the real HandoffController, independent of whatever
    control flow the loop happens to have."""
    handoff = HandoffController()
    from cua.handoff import InterventionRequest
    from cua.handoff.controller import ControlViolation

    await handoff.escalate(InterventionRequest.create(
        run_id="run-1", mode="discovery", reason_code="HUMAN_REQUESTED", reason_message="x", current_url="http://x/",
    ))
    try:
        handoff.assert_automation_control()
        assert False, "expected ControlViolation while HUMAN_CONTROL is active"
    except ControlViolation:
        pass


async def test_intervention_budget_exhaustion_stops_discovery_cleanly(tmp_path: Path):
    """Section 7: a bounded total-intervention count -- once spent, a
    further handoff-eligible stop must not hang or crash; it becomes a
    clean, typed INTERVENTION_EXHAUSTED stop."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "first"}, reasoning="x"),
        AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "second"}, reasoning="x"),
    ])
    handoff = HandoffController(max_interventions=1)
    loop = _make_loop(surface, model, tmp_path, handoff=handoff)

    async def operator_resumes_once() -> None:
        await asyncio.sleep(0)
        handoff.request_resume()

    trace, _ = await asyncio.gather(loop.run("goal"), operator_resumes_once())

    assert trace.stop_reason == DiscoveryStopReason.INTERVENTION_EXHAUSTED
    assert len(handoff.interventions) == 1  # the second escalate() never got recorded -- budget exceeded first


async def test_without_a_handoff_attached_behavior_is_unchanged(tmp_path: Path):
    """Backward compatibility (Section 9's spirit applied to discovery
    too): a DiscoveryLoop built with no handoff at all still stops
    immediately on HUMAN_REQUESTED, exactly like Phase 4."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "no handoff here"}, reasoning="need help")
    ])
    loop = _make_loop(surface, model, tmp_path)  # handoff=None

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.HUMAN_REQUESTED
    assert trace.stop_message == "no handoff here"
