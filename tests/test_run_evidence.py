"""Orchestration-level tests for scripts/run_evidence.py.

IMPORTANT: nothing in this file is genuine Phase 8 LLM evidence, and none
of it is presented as such. It exercises the runner's own plumbing --
credential handling, subprocess environment hygiene, artifact
parameterization/round-trip, and the "replay makes zero model calls"
boundary -- using FakeModelAdapter or no model at all, exactly the way
the rest of this repository's tests do everywhere a real API key isn't
available. A genuine run only ever happens via `python
scripts/run_evidence.py` with a real ANTHROPIC_API_KEY set by a human in
their own shell; that path is not exercised by pytest anywhere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_evidence  # noqa: E402
from cua.actions import ActionType  # noqa: E402
from cua.agent.actions import AgentAction  # noqa: E402
from cua.agent.evidence import DiscoveryEvidenceCapture  # noqa: E402
from cua.agent.loop import DiscoveryLoop  # noqa: E402
from cua.agent.trace import DiscoveryStopReason, DiscoveryTrace  # noqa: E402
from cua.artifact.targets import TargetDescriptor  # noqa: E402
from cua.observability.logger import StructuredLogger  # noqa: E402
from cua.results.taxonomy import RunStatus  # noqa: E402
from cua.surfaces.playwright_web import PlaywrightWebSurface  # noqa: E402

from fake_model import FakeModelAdapter  # noqa: E402


# -- credential handling: never accepts a key any way but the environment --

def test_require_api_key_fails_clearly_when_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(run_evidence.EvidenceRunFailure, match="ANTHROPIC_API_KEY is not set"):
        run_evidence._require_api_key()


def test_require_api_key_reads_only_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-real-key")
    assert run_evidence._require_api_key() == "test-placeholder-not-a-real-key"


def test_require_model_name_fails_clearly_when_absent(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(run_evidence.EvidenceRunFailure, match="MODEL_NAME is not set"):
        run_evidence._require_model_name()


def test_require_model_name_returns_the_configured_value(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "claude-test-model")
    assert run_evidence._require_model_name() == "claude-test-model"


def test_main_exits_nonzero_and_never_starts_the_demo_app_without_a_key(monkeypatch, capsys):
    """The very first thing main() must do is fail on a missing key --
    before spawning any subprocess or launching a browser."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    started = {"called": False}
    monkeypatch.setattr(run_evidence, "_start_demo_app", lambda: started.__setitem__("called", True))

    exit_code = run_evidence.main()

    assert exit_code == 1
    assert started["called"] is False
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "test-placeholder" not in captured.err  # no stray key-shaped value ever printed


# -- subprocess environment hygiene: the demo app never sees the key --

def test_demo_app_subprocess_env_excludes_the_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-never-reach-the-demo-app")
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")

    env = run_evidence._demo_app_subprocess_env()

    assert "ANTHROPIC_API_KEY" not in env
    assert env.get("SOME_OTHER_VAR") == "kept"
    assert env.get("PYTHONPATH") == str(REPO_ROOT / "src")


def test_demo_app_subprocess_env_is_safe_even_when_key_is_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = run_evidence._demo_app_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env


# -- failure-evidence hygiene: sanitized, no chain-of-thought/prompt content --

def test_preserve_discovery_failure_writes_only_sanitized_summary_fields(tmp_path: Path):
    trace = DiscoveryTrace(run_id="fake-run-id", goal=run_evidence.GOAL)
    trace.stop_reason = DiscoveryStopReason.MAX_STEPS
    trace.stop_message = "reached max_steps=15 without FINISH"
    model = FakeModelAdapter([])
    model.calls.append((object(), []))  # simulate one recorded decision call

    failure_dir = tmp_path / "discovery_failure"
    run_evidence._preserve_discovery_failure(trace, model, discovery_failure_dir=failure_dir)

    written = json.loads((failure_dir / "discovery_failure_summary.json").read_text())
    assert written["stop_reason"] == "MAX_STEPS"
    assert written["stop_message"] == "reached max_steps=15 without FINISH"
    assert written["genuine_model_decision_calls"] == 1
    assert written["run_id"] == "fake-run-id"
    # only the known-safe summary keys -- never a raw transcript/prompt field
    assert set(written) == {
        "goal", "stop_reason", "stop_message", "steps_executed",
        "genuine_model_decision_calls", "run_id", "note",
    }


# -- artifact parameterization + round-trip + replay zero-model-call boundary --
# These use a real Chromium session and the real demo app (via the shared
# conftest.py fixture) with a FakeModelAdapter standing in for the model
# ONLY to produce a DiscoveryTrace to feed the runner's own artifact/replay
# plumbing -- never presented as genuine evidence (see module docstring).

def _scripted_discovery_actions(member_id: str, base_url: str) -> list[AgentAction]:
    return [
        AgentAction(action=ActionType.NAVIGATE, params={"url": f"{base_url}/search"}, reasoning="go to search"),
        AgentAction(action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
                    params={"value": member_id}, reasoning="enter member id"),
        AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"),
                    reasoning="submit"),
        AgentAction(action=ActionType.CHECKPOINT, params={"description": "results page reached"}, reasoning="confirm results"),
        AgentAction(action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
                    params={"output_name": "savings_balance"}, reasoning="read balance"),
        AgentAction(action=ActionType.FINISH, reasoning="done"),
    ]


async def _fake_discovery_trace(base_url: str, tmp_path: Path) -> DiscoveryTrace:
    model = FakeModelAdapter(_scripted_discovery_actions(run_evidence.DISCOVERY_MEMBER_ID, base_url))
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH"))
    logger = StructuredLogger(run_id="test-discovery", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    loop = DiscoveryLoop(model, surface, run_evidence._policy(), logger, evidence, max_steps=10)
    try:
        return await loop.run(run_evidence.GOAL)
    finally:
        await surface.close()


async def test_build_and_save_artifact_parameterizes_and_round_trips(tmp_path: Path, demo_app_base_url: str):
    trace = await _fake_discovery_trace(demo_app_base_url, tmp_path)
    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED

    evidence_dir = tmp_path / "evidence_out"
    artifact, saved_path = run_evidence._build_and_save_artifact(trace, demo_app_base_url, evidence_dir=evidence_dir)

    assert saved_path.exists()
    assert saved_path.name == f"{run_evidence.CAPABILITY_ID}@{run_evidence.CAPABILITY_VERSION}.yaml"
    fill_step = next(s for s in artifact.steps if s.action == ActionType.FILL)
    assert fill_step.params["value"] == "{{member_id}}"
    # the literal discovery value may legitimately appear once, as the
    # declared InputParam's illustrative `example` -- but never as a FILL
    # step's actual params['value'], which is what would mean the artifact
    # replays only the one literal id it happened to discover with.
    saved_yaml = saved_path.read_text()
    assert "value: '{{member_id}}'" in saved_yaml or 'value: "{{member_id}}"' in saved_yaml or "value: {{member_id}}" in saved_yaml


async def test_build_and_save_artifact_refuses_a_non_parameterized_artifact(monkeypatch, tmp_path: Path, demo_app_base_url: str):
    """If ArtifactBuilder ever produced a FILL step that did NOT get
    parameterized (a regression this runner must catch, not silently
    persist), _build_and_save_artifact must raise rather than save."""
    trace = await _fake_discovery_trace(demo_app_base_url, tmp_path)
    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED

    real_builder_cls = run_evidence.ArtifactBuilder  # captured before patching, to avoid self-recursion

    class _BadBuilder:
        def build(self, *args, **kwargs):
            real = real_builder_cls().build(*args, **kwargs)
            for step in real.steps:
                if step.action == ActionType.FILL:
                    step.params["value"] = run_evidence.DISCOVERY_MEMBER_ID  # un-parameterize it
            return real

    monkeypatch.setattr(run_evidence, "ArtifactBuilder", _BadBuilder)
    with pytest.raises(run_evidence.EvidenceRunFailure, match="did not parameterize"):
        run_evidence._build_and_save_artifact(trace, demo_app_base_url, evidence_dir=tmp_path / "evidence_out")


async def test_run_replay_makes_zero_model_calls_and_uses_a_fresh_surface(tmp_path: Path, demo_app_base_url: str):
    """Replay must never call model.decide() -- proven here by handing it
    a FakeModelAdapter with an EMPTY script, which raises immediately if
    anything ever tries to call it, not merely by comparing counts."""
    trace = await _fake_discovery_trace(demo_app_base_url, tmp_path)
    artifact, _ = run_evidence._build_and_save_artifact(trace, demo_app_base_url, evidence_dir=tmp_path / "evidence_out")

    empty_model = FakeModelAdapter([])  # any decide() call would raise ModelOutputError: script exhausted
    replay_dir = tmp_path / "replay_out"
    result, calls_before, calls_after = await run_evidence._run_replay(artifact, empty_model, replay_evidence_dir=replay_dir)

    assert result.status == RunStatus.SUCCESS
    assert result.outputs == {"savings_balance": 542.10}
    assert calls_before == 0 and calls_after == 0
    assert len(empty_model.calls) == 0
    assert (replay_dir / "replay_success.png").exists()
