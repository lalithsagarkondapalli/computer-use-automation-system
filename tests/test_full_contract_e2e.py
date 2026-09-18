"""Phase 7's centerpiece: the complete architecture boundary-to-boundary,
model-independent, against the REAL demo app and REAL Chromium throughout
-- no FakeSurfaceAdapter anywhere in this file (that's what distinguishes
this from Phase 4's tests/test_discovery_to_replay_integration.py, which
proves the same discovery->artifact->replay contract but entirely against
FakeSurfaceAdapter).

    natural-language goal
    -> FakeModelAdapter (deterministic, scripted -- zero API key, zero
       network model calls) driving DiscoveryLoop against a real
       PlaywrightWebSurface + the real running demo app
    -> a successful DiscoveryTrace (GOAL_ACHIEVED)
    -> ArtifactBuilder -> a real CapabilityArtifact
    -> saved to real YAML on disk (cua.artifact.store.save_artifact)
    -> loaded back (cua.artifact.store.load_artifact) -- no manual
       correction of the YAML in between
    -> a FRESH PlaywrightWebSurface (a second, independent browser
       instance/session -- proving the artifact is portable, not tied to
       discovery's own session)
    -> ReplayEngine, with a DIFFERENT valid member_id than discovery used
    -> RunStatus.SUCCESS with a typed, correctly-parsed savings_balance
    -> and the FakeModelAdapter's call count during the REPLAY phase is
       asserted to be exactly 0 -- the strongest, most literal form of
       "zero model calls during replay" this suite can assert.
"""
from __future__ import annotations

import os
from pathlib import Path

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.agent.artifact_builder import ArtifactBuilder
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.agent.trace import DiscoveryStopReason
from cua.artifact.schema import InputParam, OutputParam, ParamType
from cua.artifact.store import load_artifact, save_artifact
from cua.artifact.targets import TargetDescriptor
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.surfaces.playwright_web import PlaywrightWebSurface

from fake_model import FakeModelAdapter

_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")


def _policy() -> PolicyEngine:
    return PolicyEngine(PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                          ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT),
    ))


def _discovery_script(member_id: str, base_url: str) -> list[AgentAction]:
    """The exact, deterministic action sequence a human reviewer would
    also take to look up a member on the real demo app -- scripted here
    instead of coming from a real model call, so this test needs no API
    key, but every action still passes through DiscoveryLoop's real
    VALIDATE/POLICY CHECK against what the real page actually shows.

    The very first NAVIGATE uses an absolute URL, not a relative route --
    a freshly-launched surface starts at about:blank, so there is no
    "current page" for a relative route to be resolved against yet (that
    is genuinely what a model exploring a brand-new app would have to do
    too: it cannot know a relative path is meaningful until something is
    already loaded). Every subsequent NAVIGATE in a real discovered
    capability is free to use a relative route once a page is loaded --
    see cua.artifact.targets/ReplayEngine, which resolves those against
    the artifact's own recorded target.base_url_pattern instead.
    """
    return [
        AgentAction(action=ActionType.NAVIGATE, params={"url": f"{base_url}/search"}, reasoning="Go to the member search page."),
        AgentAction(
            action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
            params={"value": member_id}, reasoning="Enter the member id.",
        ),
        AgentAction(
            action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"),
            reasoning="Submit the search.",
        ),
        AgentAction(action=ActionType.CHECKPOINT, params={"description": "results page reached"}, reasoning="Confirm we reached the results page."),
        AgentAction(
            action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
            params={"output_name": "savings_balance"}, reasoning="Read the savings balance.",
        ),
        AgentAction(action=ActionType.FINISH, reasoning="Found the balance; goal achieved."),
    ]


async def test_full_contract_discovery_to_replay_against_real_chromium_and_real_demo_app(tmp_path: Path, demo_app_base_url: str):
    discovery_member_id = "12345"
    replay_member_id = "67890"  # deliberately DIFFERENT from what discovery observed

    # -- 1. DISCOVERY: FakeModelAdapter (deterministic, no API key) driving
    # a REAL PlaywrightWebSurface against the REAL running demo app --
    model = FakeModelAdapter(list(_discovery_script(discovery_member_id, demo_app_base_url)))
    discovery_surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    discovery_logger = StructuredLogger(run_id="contract-discovery", log_dir=tmp_path / "logs")
    discovery_evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, discovery_surface, _policy(), discovery_logger, discovery_evidence, max_steps=10)

    try:
        trace = await loop.run(f"Look up member {discovery_member_id} and report their savings balance.")
    finally:
        await discovery_surface.close()

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert len(model.calls) == 6  # exactly one decide() per scripted step -- proves the model WAS actually used here

    # -- 2. ARTIFACT BUILDER: trace -> a real, schema-validated CapabilityArtifact --
    builder = ArtifactBuilder()
    artifact = builder.build(
        trace,
        capability_id="member_lookup_balance",
        capability_version="7.0.0-contract-test",
        name="Look up member and read savings balance (Phase 7 contract test)",
        description="Built from a real discovery run against the real demo app, for the Phase 7 full-contract test.",
        app_name="CU Ops Console",
        base_url_pattern=demo_app_base_url,
        inputs=[InputParam(name="member_id", type=ParamType.STRING, description="5-digit synthetic member identifier", example=discovery_member_id)],
        outputs=[OutputParam(name="savings_balance", type=ParamType.FLOAT, description="Current savings balance")],
        parameter_values={"member_id": discovery_member_id},
    )
    assert artifact.capability_id == "member_lookup_balance"
    # the discovered artifact must be genuinely parameterized -- the literal
    # "12345" discovery happened to type must NOT be baked into the artifact
    fill_step = next(s for s in artifact.steps if s.action == ActionType.FILL)
    assert fill_step.params["value"] == "{{member_id}}"

    # -- 3. ROUND-TRIP: save real YAML, load it back, no manual correction --
    saved_path = save_artifact(artifact, tmp_path / "artifacts")
    assert saved_path.exists()
    loaded_artifact = load_artifact(saved_path)
    assert loaded_artifact == artifact  # full semantic equivalence, not just "it loaded"

    # -- 4. REPLAY: a FRESH, independent browser session, a DIFFERENT valid
    # input than discovery used, zero model involvement --
    replay_surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    assert replay_surface is not discovery_surface  # genuinely a second session, not reused
    replay_logger = StructuredLogger(run_id="contract-replay", log_dir=tmp_path / "logs")
    replay_evidence = EvidenceCapture(tmp_path / "evidence")
    engine = ReplayEngine(_policy(), replay_logger, replay_evidence)

    try:
        result = await engine.run(loaded_artifact, {"member_id": replay_member_id}, replay_surface)
    finally:
        await replay_surface.close()

    assert result.status == RunStatus.SUCCESS
    assert result.code is None
    assert result.outputs == {"savings_balance": 542.10}  # SYNTH-MEMBER-DELTA's balance -- a value discovery never saw

    # -- the literal assertion Phase 7 asked for: zero model calls during replay --
    assert len(model.calls) == 6  # unchanged since discovery finished -- replay never touched the model adapter at all
