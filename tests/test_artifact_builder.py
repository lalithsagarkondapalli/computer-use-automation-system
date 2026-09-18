"""ArtifactBuilder tests: does a real DiscoveryTrace (produced by actually
running DiscoveryLoop against FakeSurfaceAdapter/FakeModelAdapter) turn
into a valid, replay-ready CapabilityArtifact -- with discovery-only
actions stripped, runtime values parameterized, and outputs/checkpoints
correctly typed. No hand-written trace objects: every trace here comes
from a real `await loop.run(...)` call, same as test_discovery_loop.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cua.actions import ActionType
from cua.agent.artifact_builder import ArtifactBuilder, ArtifactBuildError
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.artifact.checks import ConditionProbe, TextPresentCheck
from cua.artifact.schema import InputParam, OutputParam, ParamType, RiskLevel
from cua.observability.logger import StructuredLogger
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig

from discovery_fixtures import build_member_lookup_fixture

_INPUTS = [InputParam(name="member_id", type=ParamType.STRING, description="Member identifier", example="12345")]
_OUTPUTS = [OutputParam(name="savings_balance", type=ParamType.STRING, description="Current savings balance")]


async def _run_member_lookup(tmp_path: Path, **fixture_overrides):
    surface, model = build_member_lookup_fixture(**fixture_overrides)
    policy = PolicyEngine(
        PolicyConfig(
            allowed_domains=("127.0.0.1",),
            allowed_actions=(
                ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT,
            ),
        )
    )
    logger = StructuredLogger(run_id="artifact-builder-test", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, policy, logger, evidence, max_steps=10)
    return await loop.run("Look up a member and report their savings balance.")


async def test_builder_produces_a_valid_replay_ready_artifact(tmp_path: Path):
    trace = await _run_member_lookup(tmp_path, member_id="12345", balance="$1,234.56")

    artifact = ArtifactBuilder().build(
        trace,
        capability_id="member_lookup_balance",
        capability_version="1.0.0",
        name="Look up member and read savings balance",
        description="Search for a member by id and extract their current savings balance.",
        app_name="CU Ops Console",
        base_url_pattern="http://127.0.0.1:8000",
        inputs=_INPUTS,
        outputs=_OUTPUTS,
        parameter_values={"member_id": "12345"},
    )

    actions = [step.action for step in artifact.steps]
    assert actions == [
        ActionType.NAVIGATE, ActionType.FILL, ActionType.CLICK, ActionType.CHECKPOINT, ActionType.READ_EXTRACT,
    ]
    assert ActionType.FINISH not in actions and ActionType.REQUEST_HUMAN not in actions

    fill_step = next(s for s in artifact.steps if s.action == ActionType.FILL)
    assert fill_step.params["value"] == "{{member_id}}"  # the literal "12345" was parameterized, not baked in

    extract_step = next(s for s in artifact.steps if s.action == ActionType.READ_EXTRACT)
    assert extract_step.extraction.output_name == "savings_balance"
    assert extract_step.extraction.transform == "currency_to_float"  # inferred from the observed "$1,234.56" shape

    checkpoint_step = next(s for s in artifact.steps if s.action == ActionType.CHECKPOINT)
    assert checkpoint_step.checkpoint.assertion.type == "url_matches"
    assert checkpoint_step.checkpoint.assertion.pattern == "/results"

    assert artifact.risk == RiskLevel.SAFE
    assert artifact.discovery_run_id == trace.run_id
    assert artifact.created_by == "discovery_agent"


async def test_builder_rejects_a_trace_that_did_not_reach_goal_achieved(tmp_path: Path):
    from cua.agent.actions import AgentAction
    from fake_model import FakeModelAdapter
    from fakes import FakeSurfaceAdapter
    from discovery_fixtures import obs

    surface = FakeSurfaceAdapter()
    surface.script_observation(obs("http://127.0.0.1:8000/", []))
    model = FakeModelAdapter([AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "stuck"}, reasoning="need help")])
    policy = PolicyEngine(PolicyConfig(allowed_domains=("127.0.0.1",), allowed_actions=tuple(ActionType)))
    logger = StructuredLogger(run_id="incomplete", log_dir=tmp_path / "logs")
    loop = DiscoveryLoop(model, surface, policy, logger, max_steps=5)
    trace = await loop.run("goal")

    with pytest.raises(ArtifactBuildError, match="did not reach GOAL_ACHIEVED"):
        ArtifactBuilder().build(
            trace, capability_id="x", capability_version="1.0.0", name="x", description="x",
            app_name="x", base_url_pattern="http://127.0.0.1:8000", inputs=[], outputs=[],
        )


async def test_builder_rejects_an_undeclared_output(tmp_path: Path):
    trace = await _run_member_lookup(tmp_path)

    with pytest.raises(ArtifactBuildError, match="not declared in `outputs`"):
        ArtifactBuilder().build(
            trace, capability_id="member_lookup_balance", capability_version="1.0.0",
            name="x", description="x", app_name="CU Ops Console", base_url_pattern="http://127.0.0.1:8000",
            inputs=_INPUTS, outputs=[],  # savings_balance never declared
            parameter_values={"member_id": "12345"},
        )


async def test_builder_attaches_known_business_outcome_probes_to_the_last_checkpoint(tmp_path: Path):
    trace = await _run_member_lookup(tmp_path)
    probe = ConditionProbe(
        description="not-found message shown",
        assertion=TextPresentCheck(text="No member found"),
        status=RunStatus.BUSINESS_OUTCOME,
        code=OutcomeCode.CUSTOMER_NOT_FOUND,
    )

    artifact = ArtifactBuilder().build(
        trace, capability_id="member_lookup_balance", capability_version="1.0.0",
        name="x", description="x", app_name="CU Ops Console", base_url_pattern="http://127.0.0.1:8000",
        inputs=_INPUTS, outputs=_OUTPUTS, parameter_values={"member_id": "12345"},
        known_business_outcome_probes=[probe],
    )

    checkpoint_step = next(s for s in artifact.steps if s.action == ActionType.CHECKPOINT)
    assert len(checkpoint_step.checkpoint.probes) == 1
    assert checkpoint_step.checkpoint.probes[0].code == OutcomeCode.CUSTOMER_NOT_FOUND


async def test_builder_redacts_secret_shaped_values_even_if_not_parameterized(tmp_path: Path):
    """Defense in depth: even a literal value that parameterization
    doesn't catch (e.g. it wasn't declared as an input) must not reach
    the artifact in the clear if it's secret-shaped."""
    from cua.agent.actions import AgentAction
    from cua.artifact.targets import TargetDescriptor
    from cua.surfaces.observation import InteractiveElement
    from fake_model import FakeModelAdapter
    from fakes import FakeElement, FakeSurfaceAdapter
    from discovery_fixtures import BASE, obs

    surface = FakeSurfaceAdapter()
    token_el = InteractiveElement(role="textbox", accessible_name="Token")
    before_fill = obs(f"{BASE}/", [token_el])
    after_fill = obs(f"{BASE}/", [token_el], "token entered successfully")
    surface.script_observation(before_fill)
    surface.script_observation(after_fill)
    surface.script_observation(after_fill)
    surface.register_role_name("textbox", "Token", [FakeElement()])
    model = FakeModelAdapter([
        AgentAction(
            action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Token"),
            params={"value": "api_key=sk-should-not-be-persisted"}, reasoning="fill it",
        ),
        AgentAction(action=ActionType.CHECKPOINT, params={"description": "token entered"}, reasoning="confirm entry"),
        AgentAction(action=ActionType.FINISH, reasoning="done"),
    ])
    policy = PolicyEngine(PolicyConfig(allowed_domains=("127.0.0.1",), allowed_actions=tuple(ActionType)))
    logger = StructuredLogger(run_id="redact-test", log_dir=tmp_path / "logs")
    loop = DiscoveryLoop(model, surface, policy, logger, max_steps=5)
    trace = await loop.run("goal")

    artifact = ArtifactBuilder().build(
        trace, capability_id="x", capability_version="1.0.0", name="x", description="x",
        app_name="x", base_url_pattern=BASE, inputs=[], outputs=[], parameter_values={},
    )

    fill_step = next(s for s in artifact.steps if s.action == ActionType.FILL)
    assert "sk-should-not-be-persisted" not in fill_step.params["value"]
    assert "[REDACTED]" in fill_step.params["value"]
