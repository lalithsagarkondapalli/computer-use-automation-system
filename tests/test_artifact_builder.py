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
from cua.agent.actions import AgentAction
from cua.agent.artifact_builder import ArtifactBuilder, ArtifactBuildError
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.artifact.checks import ConditionProbe, TextPresentCheck
from cua.artifact.schema import InputParam, OutputParam, ParamType, RiskLevel
from cua.artifact.store import load_artifact, save_artifact
from cua.artifact.targets import TargetDescriptor
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.surfaces.observation import InteractiveElement

from discovery_fixtures import BASE, build_member_lookup_fixture, obs
from fake_model import FakeModelAdapter
from fakes import FakeElement, FakeSurfaceAdapter

_INPUTS = [InputParam(name="member_id", type=ParamType.STRING, description="Member identifier", example="12345")]
_OUTPUTS = [OutputParam(name="savings_balance", type=ParamType.STRING, description="Current savings balance")]


def _policy() -> PolicyEngine:
    return PolicyEngine(
        PolicyConfig(
            allowed_domains=("127.0.0.1",),
            allowed_actions=(
                ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT,
            ),
        )
    )


def _build_member_lookup_fixture_no_explicit_checkpoint(
    *, member_id: str = "12345", balance: str = "$1,234.56"
) -> tuple[FakeSurfaceAdapter, FakeModelAdapter]:
    """The same member_lookup_balance happy path `discovery_fixtures.build_member_lookup_fixture`
    scripts, minus the explicit CHECKPOINT action -- i.e. exactly the shape a
    genuine model can produce when it takes the shortest path from
    extraction straight to FINISH without ever proposing CHECKPOINT itself
    (the real Phase 8 failure this fix addresses). Kept local to this test
    module, deliberately not folded into `discovery_fixtures.py`, since that
    module's own docstring scopes it to the hand-authored-vs-discovered
    comparison in test_discovery_to_replay_integration.py."""
    surface = FakeSurfaceAdapter()
    member_id_el = InteractiveElement(role="textbox", accessible_name="Member ID")
    search_el = InteractiveElement(role="button", accessible_name="Search")
    balance_el = InteractiveElement(role="status", accessible_name="Savings Balance")

    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.register_role_name("status", "Savings Balance", [FakeElement(text=balance)])

    obs0 = obs(f"{BASE}/", [], "Welcome to CU Ops Console")
    obs1 = obs(f"{BASE}/search", [member_id_el, search_el], "Member ID Search")
    obs2 = obs1
    obs3 = obs(f"{BASE}/results", [balance_el], f"Savings Balance {balance}")
    obs4 = obs3
    for o in (obs0, obs1, obs2, obs3, obs4):
        surface.script_observation(o)

    script = [
        AgentAction(action=ActionType.NAVIGATE, params={"route": "/search"}, reasoning="Go to the search page."),
        AgentAction(
            action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
            params={"value": member_id}, reasoning="Enter the member id.",
        ),
        AgentAction(
            action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"),
            reasoning="Submit the search.",
        ),
        AgentAction(
            action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
            params={"output_name": "savings_balance"}, reasoning="Read the balance.",
        ),
        AgentAction(action=ActionType.FINISH, reasoning="Found the balance; goal achieved -- no CHECKPOINT proposed."),
    ]
    model = FakeModelAdapter(script)
    return surface, model


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


# -- Phase 8 checkpoint-synthesis fix -----------------------------------
#
# A genuine model can reach GOAL_ACHIEVED without ever proposing a
# CHECKPOINT action itself (nothing in the action schema or discovery
# prompt requires it to). The tests below prove ArtifactBuilder now
# deterministically re-derives a grounded checkpoint from the trace's own
# structured data in that case -- never by trusting a bare FINISH, never
# by fabricating an observation the trace doesn't contain, and never by
# special-casing this repository's one demo app.


async def test_grounded_read_extract_without_explicit_checkpoint_synthesizes_a_valid_checkpoint(tmp_path: Path):
    """(1) The core fix: a trace whose last grounded action is a successful
    READ_EXTRACT, with no CHECKPOINT ever proposed, still builds -- and the
    synthesized checkpoint is a deterministic, verifiable condition (the
    extracted element's visibility), not a generic placeholder."""
    surface, model = _build_member_lookup_fixture_no_explicit_checkpoint()
    logger = StructuredLogger(run_id="synth-basic", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, _policy(), logger, evidence, max_steps=10)
    trace = await loop.run("Look up member 12345 and report their savings balance.")

    assert trace.stop_reason.value == "GOAL_ACHIEVED"
    assert not any(r.action is not None and r.action.action == ActionType.CHECKPOINT for r in trace.steps)

    artifact = ArtifactBuilder().build(
        trace,
        capability_id="member_lookup_balance",
        capability_version="1.0.0",
        name="Look up member and read savings balance",
        description="Search for a member by id and extract their current savings balance.",
        app_name="CU Ops Console",
        base_url_pattern=BASE,
        inputs=_INPUTS,
        outputs=_OUTPUTS,
        parameter_values={"member_id": "12345"},
    )

    checkpoint_steps = [s for s in artifact.steps if s.action == ActionType.CHECKPOINT]
    assert len(checkpoint_steps) == 1
    checkpoint = checkpoint_steps[0].checkpoint
    assert checkpoint.assertion.type == "element_visible"
    assert checkpoint.assertion.target.role == "status"
    assert checkpoint.assertion.target.accessible_name == "Savings Balance"

    # inserted immediately before the READ_EXTRACT it was derived from,
    # matching the convention a hand-authored/model-emitted checkpoint uses
    actions = [s.action for s in artifact.steps]
    read_index = actions.index(ActionType.READ_EXTRACT)
    assert actions[read_index - 1] == ActionType.CHECKPOINT


async def test_explicit_checkpoint_is_preserved_and_not_duplicated(tmp_path: Path):
    """(2) When the model DOES propose CHECKPOINT itself, synthesis must
    not run at all -- the artifact keeps exactly the model's own checkpoint,
    never an extra synthesized one alongside it."""
    trace = await _run_member_lookup(tmp_path)

    artifact = ArtifactBuilder().build(
        trace, capability_id="member_lookup_balance", capability_version="1.0.0",
        name="x", description="x", app_name="CU Ops Console", base_url_pattern=BASE,
        inputs=_INPUTS, outputs=_OUTPUTS, parameter_values={"member_id": "12345"},
    )

    checkpoint_steps = [s for s in artifact.steps if s.action == ActionType.CHECKPOINT]
    assert len(checkpoint_steps) == 1
    assert checkpoint_steps[0].checkpoint.assertion.type == "url_matches"
    assert checkpoint_steps[0].checkpoint.assertion.pattern == "/results"  # the model's own claim, untouched


async def test_finish_with_no_grounded_evidence_fails_safely(tmp_path: Path):
    """(3) A trace that reaches FINISH having never extracted anything and
    never observably changed state must be REJECTED, not padded with a
    fabricated checkpoint -- bare FINISH is never treated as proof."""
    surface = FakeSurfaceAdapter()
    surface.register_role_name("status", "Ready", [FakeElement()])  # so WAIT resolves and executes
    # Grounding (VALIDATE) requires WAIT's target to already be present in
    # the observation it's proposed against -- script the identical
    # observation both before and after the WAIT so nothing observably
    # changes across it, matching the scenario this test needs: an
    # executed, persistable step that nonetheless offers no groundable
    # evidence of success.
    ready_observation = obs(f"{BASE}/", [InteractiveElement(role="status", accessible_name="Ready")], "page ready")
    surface.script_observation(ready_observation)
    surface.script_observation(ready_observation)
    model = FakeModelAdapter([
        AgentAction(
            action=ActionType.WAIT, target=TargetDescriptor(role="status", accessible_name="Ready"),
            params={}, reasoning="wait for the page to be ready",
        ),
        AgentAction(action=ActionType.FINISH, reasoning="done -- nothing was ever extracted or observed to change"),
    ])
    policy = PolicyEngine(PolicyConfig(allowed_domains=("127.0.0.1",), allowed_actions=tuple(ActionType)))
    logger = StructuredLogger(run_id="no-evidence", log_dir=tmp_path / "logs")
    loop = DiscoveryLoop(model, surface, policy, logger, max_steps=5)
    trace = await loop.run("goal")
    assert trace.stop_reason.value == "GOAL_ACHIEVED"
    executed_actions = [r.action.action for r in trace.steps if r.executed and r.action is not None]
    assert executed_actions == [ActionType.WAIT, ActionType.FINISH]  # sanity: WAIT really did execute, this isn't testing an empty trace

    with pytest.raises(ArtifactBuildError, match="no groundable evidence"):
        ArtifactBuilder().build(
            trace, capability_id="x", capability_version="1.0.0", name="x", description="x",
            app_name="x", base_url_pattern=BASE, inputs=[], outputs=[],
        )


async def test_synthesized_checkpoint_is_replay_verifiable_without_model_calls(tmp_path: Path):
    """(4) The synthesized checkpoint isn't just schema-valid -- a fresh
    ReplayEngine run, against a second independent surface with no
    discovery/model code involved, actually verifies it and succeeds."""
    surface, model = _build_member_lookup_fixture_no_explicit_checkpoint()
    logger = StructuredLogger(run_id="synth-replay-discovery", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, _policy(), logger, evidence, max_steps=10)
    trace = await loop.run("Look up member 12345 and report their savings balance.")

    artifact = ArtifactBuilder().build(
        trace, capability_id="member_lookup_balance", capability_version="1.0.0",
        name="Look up member and read savings balance",
        description="Search for a member by id and extract their current savings balance.",
        app_name="CU Ops Console", base_url_pattern=BASE,
        inputs=_INPUTS, outputs=_OUTPUTS, parameter_values={"member_id": "12345"},
    )
    discovery_call_count = len(model.calls)
    assert discovery_call_count > 0

    replay_surface = FakeSurfaceAdapter()
    replay_surface.register_role_name("textbox", "Member ID", [FakeElement()])
    replay_surface.register_role_name("button", "Search", [FakeElement()])
    replay_surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    # ElementVisibleCheck has no `pattern`/`text` attribute, so
    # FakeSurfaceAdapter.wait_for_checkpoint keys it as "unknown" (see
    # fakes.py) -- the real PlaywrightWebSurface instead genuinely resolves
    # and polls the target's live visibility (see test_playwright_web_surface.py).
    replay_surface.script_checkpoint("unknown", True)

    engine = ReplayEngine(
        _policy(), StructuredLogger(run_id="synth-replay", log_dir=tmp_path / "logs"), EvidenceCapture(tmp_path / "evidence")
    )
    result = await engine.run(artifact, {"member_id": "12345"}, replay_surface)

    assert result.status == RunStatus.SUCCESS
    assert result.outputs == {"savings_balance": 1234.56}
    assert len(model.calls) == discovery_call_count  # replay never called the discovery model


async def test_checkpoint_synthesis_is_not_hardcoded_to_the_member_lookup_demo(tmp_path: Path):
    """(5) A structurally different capability -- an inventory/SKU price
    lookup, nothing to do with members or savings balances -- proves
    synthesis is driven only by the trace's own READ_EXTRACT target, never
    special-cased for this repository's one demo app."""
    surface = FakeSurfaceAdapter()
    sku_el = InteractiveElement(role="textbox", accessible_name="SKU")
    lookup_el = InteractiveElement(role="button", accessible_name="Lookup")
    price_el = InteractiveElement(role="status", accessible_name="Unit Price")

    surface.register_role_name("textbox", "SKU", [FakeElement()])
    surface.register_role_name("button", "Lookup", [FakeElement()])
    surface.register_role_name("status", "Unit Price", [FakeElement(text="$9.99")])

    obs0 = obs(f"{BASE}/", [], "Inventory System")
    obs1 = obs(f"{BASE}/inventory", [sku_el, lookup_el], "SKU Lookup")
    obs2 = obs1
    obs3 = obs(f"{BASE}/inventory/result", [price_el], "Unit Price $9.99")
    obs4 = obs3
    for o in (obs0, obs1, obs2, obs3, obs4):
        surface.script_observation(o)

    model = FakeModelAdapter([
        AgentAction(action=ActionType.NAVIGATE, params={"route": "/inventory"}, reasoning="go to inventory lookup"),
        AgentAction(
            action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="SKU"),
            params={"value": "SKU-1"}, reasoning="enter the sku",
        ),
        AgentAction(
            action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Lookup"),
            reasoning="submit the lookup",
        ),
        AgentAction(
            action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Unit Price"),
            params={"output_name": "unit_price"}, reasoning="read the unit price",
        ),
        AgentAction(action=ActionType.FINISH, reasoning="found the price; goal achieved"),
    ])
    policy = PolicyEngine(PolicyConfig(allowed_domains=("127.0.0.1",), allowed_actions=tuple(ActionType)))
    logger = StructuredLogger(run_id="no-hardcoding", log_dir=tmp_path / "logs")
    loop = DiscoveryLoop(model, surface, policy, logger, max_steps=10)
    trace = await loop.run("Look up SKU-1 and report its unit price.")
    assert trace.stop_reason.value == "GOAL_ACHIEVED"

    artifact = ArtifactBuilder().build(
        trace, capability_id="inventory_price_lookup", capability_version="1.0.0",
        name="Look up SKU and read unit price", description="Search inventory by SKU and extract unit price.",
        app_name="Inventory System", base_url_pattern=BASE,
        inputs=[InputParam(name="sku", type=ParamType.STRING, description="SKU", example="SKU-1")],
        outputs=[OutputParam(name="unit_price", type=ParamType.STRING, description="Unit price")],
        parameter_values={"sku": "SKU-1"},
    )

    checkpoint_steps = [s for s in artifact.steps if s.action == ActionType.CHECKPOINT]
    assert len(checkpoint_steps) == 1
    assert checkpoint_steps[0].checkpoint.assertion.type == "element_visible"
    assert checkpoint_steps[0].checkpoint.assertion.target.accessible_name == "Unit Price"


def test_capability_artifact_schema_still_rejects_steps_with_no_checkpoint():
    """(6) Direct proof the underlying schema invariant itself was NOT
    weakened by this fix: constructing a CapabilityArtifact by hand (fully
    bypassing ArtifactBuilder) with a steps list that has no CHECKPOINT
    still fails exactly as before."""
    from pydantic import ValidationError

    from cua.artifact.schema import CapabilityArtifact, Step, TargetIdentity

    with pytest.raises(ValidationError, match="at least one checkpoint"):
        CapabilityArtifact(
            capability_id="x", capability_version="1.0.0", name="x", description="x",
            target=TargetIdentity(app_name="x", base_url_pattern=BASE),
            risk=RiskLevel.SAFE, inputs=[], outputs=[],
            steps=[Step(step_id="s1", action=ActionType.NAVIGATE, params={"route": "/"}, risk=RiskLevel.SAFE)],
            created_by="test",
        )


async def test_full_discovery_to_artifact_to_replay_round_trip_with_synthesized_checkpoint(tmp_path: Path):
    """(7) The end-to-end pipeline (discovery -> ArtifactBuilder ->
    serialization round-trip -> fresh ReplayEngine run) still works with a
    synthesized checkpoint, not just the explicit-checkpoint path
    test_discovery_to_replay_integration.py already covers."""
    surface, model = _build_member_lookup_fixture_no_explicit_checkpoint()
    logger = StructuredLogger(run_id="roundtrip-discovery", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, _policy(), logger, evidence, max_steps=10)
    trace = await loop.run("Look up member 12345 and report their savings balance.")
    assert trace.stop_reason.value == "GOAL_ACHIEVED"

    artifact = ArtifactBuilder().build(
        trace, capability_id="member_lookup_balance", capability_version="1.0.0",
        name="Look up member and read savings balance",
        description="Search for a member by id and extract their current savings balance.",
        app_name="CU Ops Console", base_url_pattern=BASE,
        inputs=_INPUTS, outputs=_OUTPUTS, parameter_values={"member_id": "12345"},
    )

    saved_path = save_artifact(artifact, tmp_path / "artifacts")
    reloaded = load_artifact(saved_path)
    assert reloaded == artifact  # the synthesized checkpoint survives a real save/load round trip unchanged

    replay_surface = FakeSurfaceAdapter()
    replay_surface.register_role_name("textbox", "Member ID", [FakeElement()])
    replay_surface.register_role_name("button", "Search", [FakeElement()])
    replay_surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    replay_surface.script_checkpoint("unknown", True)

    engine = ReplayEngine(
        _policy(), StructuredLogger(run_id="roundtrip-replay", log_dir=tmp_path / "logs"), EvidenceCapture(tmp_path / "evidence")
    )
    result = await engine.run(reloaded, {"member_id": "12345"}, replay_surface)

    assert result.status == RunStatus.SUCCESS
    assert result.outputs == {"savings_balance": 1234.56}


async def test_artifact_build_never_calls_the_model(tmp_path: Path):
    """(8) ArtifactBuilder.build() takes no ModelAdapter at all -- it is
    structurally incapable of making a model call. Proven behaviorally
    here too: the discovery model's call count is identical immediately
    before and immediately after build(), including in the no-explicit-
    checkpoint path that now does extra synthesis work."""
    surface, model = _build_member_lookup_fixture_no_explicit_checkpoint()
    logger = StructuredLogger(run_id="isolation-discovery", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, _policy(), logger, evidence, max_steps=10)
    trace = await loop.run("Look up member 12345 and report their savings balance.")
    calls_after_discovery = len(model.calls)
    assert calls_after_discovery > 0

    ArtifactBuilder().build(
        trace, capability_id="member_lookup_balance", capability_version="1.0.0",
        name="x", description="x", app_name="CU Ops Console", base_url_pattern=BASE,
        inputs=_INPUTS, outputs=_OUTPUTS, parameter_values={"member_id": "12345"},
    )

    assert len(model.calls) == calls_after_discovery
