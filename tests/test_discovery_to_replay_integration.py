"""THE integration test Phase 4 specifically requires (Section 6): a
discovery trace -> ArtifactBuilder -> CapabilityArtifact validation ->
ReplayEngine consumes the artifact successfully, with a deterministic
fake model so it needs no API key.

This is the concrete, executable proof that discovery and replay are
connected by a real artifact, not by convention: the artifact `discovery`
produces here is fed to `ReplayEngine` completely unmodified -- no manual
YAML edits, no hand-patched fields -- against a second, independent
FakeSurfaceAdapter standing in for a fresh browser session, exactly as a
real replay run would use a fresh Playwright session days or weeks after
the discovery run that authored the capability.
"""
from __future__ import annotations

from pathlib import Path

from cua.actions import ActionType
from cua.agent.artifact_builder import ArtifactBuilder
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.artifact.schema import InputParam, OutputParam, ParamType
from cua.artifact.store import load_artifact, save_artifact
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig

from discovery_fixtures import build_member_lookup_fixture
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


async def test_discovery_produced_artifact_replays_to_success_with_zero_llm_involvement(tmp_path: Path):
    # -- 1. DISCOVERY: a genuine (fake-model) OBSERVE/DECIDE/VALIDATE/POLICY/ACT run --
    discovery_surface, model = build_member_lookup_fixture(member_id="12345", balance="$1,234.56")
    logger = StructuredLogger(run_id="integration-discovery", log_dir=tmp_path / "logs")
    discovery_evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, discovery_surface, _policy(), logger, discovery_evidence, max_steps=10)

    trace = await loop.run("Look up member 12345 and report their savings balance.")
    assert trace.stop_reason.value == "GOAL_ACHIEVED"

    # -- 2. ARTIFACT BUILDER: trace -> typed, versioned CapabilityArtifact --
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

    # -- 3. round-trip through the same YAML store ReplayEngine's caller uses --
    # (proves the artifact is consumable exactly as saved, no in-memory-only shortcuts)
    saved_path = save_artifact(artifact, tmp_path / "artifacts")
    reloaded = load_artifact(saved_path)
    assert reloaded == artifact

    # -- 4. REPLAY: a second, independent FakeSurfaceAdapter -- standing in for a
    # fresh browser session with no discovery/model code involved at all --
    # consumes the reloaded artifact deterministically.
    replay_surface = FakeSurfaceAdapter()
    replay_surface.register_role_name("textbox", "Member ID", [FakeElement()])
    replay_surface.register_role_name("button", "Search", [FakeElement()])
    replay_surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    replay_surface.script_checkpoint("/results", True)

    engine = ReplayEngine(_policy(), StructuredLogger(run_id="integration-replay", log_dir=tmp_path / "logs"), EvidenceCapture(tmp_path / "evidence"))
    result = await engine.run(reloaded, {"member_id": "12345"}, replay_surface)

    assert result.status == RunStatus.SUCCESS
    assert result.code is None
    assert result.outputs == {"savings_balance": 1234.56}

    # -- 5. Nothing about replay touched the model layer: same guarantee
    # test_replay_model_isolation.py proves structurally, demonstrated here
    # behaviorally -- the model's FakeModelAdapter.decide() was called
    # exactly as many times as discovery took, and zero times during replay.
    assert len(model.calls) == 6  # the discovery run's own 6 steps; replay never called decide() at all
