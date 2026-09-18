#!/usr/bin/env python3
"""Phase 8 genuine evidence runner.

Performs the complete, real, model-backed Phase 8 workflow in one process:

    natural-language goal
    -> real AnthropicModelAdapter (genuine API calls -- the model decides
       every action; nothing here is pre-scripted)
    -> DiscoveryLoop
    -> PlaywrightWebSurface against real Chromium
    -> the local synthetic demo app (started and stopped by this script)
    -> GOAL_ACHIEVED
    -> ArtifactBuilder -> a real, parameterized CapabilityArtifact
    -> saved under evidence/
    -> a FRESH browser/session
    -> ReplayEngine replays the exact saved artifact with a DIFFERENT
       member id (67890)
    -> typed SUCCESS output (savings_balance = 542.10)
    -> proof the Anthropic adapter's call count did not increase during
       replay (replay makes zero model decisions)

This script is the ONLY place in the repository that is allowed to make a
genuine Anthropic API call outside a test's own explicit, sanctioned use
of a real key. FakeModelAdapter never appears in this file, and no
AgentAction is ever pre-scripted here -- see tests/test_run_evidence.py
for the orchestration-level tests that exercise this script's structure
*without* ever presenting a mocked/faked run as Phase 8 evidence.

Credential handling (non-negotiable):
  - ANTHROPIC_API_KEY is read ONLY from the process environment
    (os.environ). It is never accepted as a CLI argument, never written
    to .env, never written to a log/evidence/artifact file, and never
    passed to the demo-app subprocess (which has no use for it and does
    not need it -- it is explicitly stripped from that subprocess's
    environment below). It is never printed, formatted, or interpolated
    into any string this script produces.
  - If ANTHROPIC_API_KEY is absent, this script fails immediately and
    clearly with a nonzero exit status -- it never falls back to
    FakeModelAdapter and never fabricates evidence of a run that did not
    happen.

Usage (from the repository root, with a venv active):
    export ANTHROPIC_API_KEY='...'      # set in your own shell, never here
    export MODEL_NAME='...'             # a current Anthropic model id
    python scripts/run_evidence.py

Exit status: 0 only if every required Phase 8 condition below was
satisfied by real, executed steps. Nonzero (with a clear diagnostic, and
sanitized failure evidence preserved under evidence/) otherwise.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anthropic import AsyncAnthropic  # noqa: E402

from cua.actions import ActionType  # noqa: E402
from cua.agent.artifact_builder import ArtifactBuildError, ArtifactBuilder  # noqa: E402
from cua.agent.evidence import DiscoveryEvidenceCapture  # noqa: E402
from cua.agent.loop import DiscoveryLoop  # noqa: E402
from cua.agent.trace import DiscoveryStopReason, DiscoveryTrace  # noqa: E402
from cua.artifact.schema import InputParam, OutputParam, ParamType  # noqa: E402
from cua.artifact.store import load_artifact, save_artifact  # noqa: E402
from cua.models.anthropic_adapter import AnthropicModelAdapter  # noqa: E402
from cua.observability.logger import StructuredLogger  # noqa: E402
from cua.replay.engine import ReplayEngine  # noqa: E402
from cua.replay.evidence import EvidenceCapture  # noqa: E402
from cua.results.taxonomy import RunStatus  # noqa: E402
from cua.safety.engine import PolicyEngine  # noqa: E402
from cua.safety.models import PolicyConfig  # noqa: E402
from cua.surfaces.playwright_web import PlaywrightWebSurface  # noqa: E402

GOAL = "Look up member 12345 and return their current savings balance."
DISCOVERY_MEMBER_ID = "12345"
REPLAY_MEMBER_ID = "67890"
EXPECTED_REPLAY_BALANCE = 542.10
CAPABILITY_ID = "discovered_member_lookup_balance"
CAPABILITY_VERSION = "1.0.0"

# Optional override, e.g. for a sandbox with a pre-installed Chromium at a
# fixed path. Unset (None) on a normal machine where `python -m playwright
# install chromium` has already downloaded a browser -- Playwright then
# resolves it automatically, exactly like every other script/test in this
# repository (see tests/test_demo_app_e2e.py's identical pattern).
_PLAYWRIGHT_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")

EVIDENCE_DIR = REPO_ROOT / "evidence"
DISCOVERY_EVIDENCE_DIR = EVIDENCE_DIR / "discovery"
DISCOVERY_FAILURE_DIR = EVIDENCE_DIR / "discovery_failure"
REPLAY_EVIDENCE_DIR = EVIDENCE_DIR / "replay_success"


class EvidenceRunFailure(Exception):
    """Raised to abort the run with a clear, sanitized message and a
    nonzero exit status. Never caught to paper over a failure -- the
    top-level main() prints it and exits nonzero."""


def _require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EvidenceRunFailure(
            "ANTHROPIC_API_KEY is not set in this process's environment. "
            "This script reads it ONLY from the environment and will not "
            "accept it any other way. Set it in your current shell, e.g.:\n"
            "    export ANTHROPIC_API_KEY='your-key-here'\n"
            "then re-run this script."
        )
    return key


def _require_model_name() -> str:
    model = os.environ.get("MODEL_NAME", "").strip()
    if not model:
        raise EvidenceRunFailure(
            "MODEL_NAME is not set. Set it to a current Anthropic model id "
            "suited to structured tool use -- see "
            "https://docs.claude.com/en/docs/about-claude/models for the "
            "current list. A small/cheap model is enough for this task "
            "(one short tool-calling decision per step, a handful of "
            "steps). Example:\n"
            "    export MODEL_NAME='claude-...'\n"
        )
    return model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _demo_app_subprocess_env() -> dict[str, str]:
    """The environment for the demo-app subprocess -- ANTHROPIC_API_KEY is
    deliberately excluded even though the parent process may have it set.
    The demo app has no use for the key and this is one more place it is
    guaranteed not to leak into. Pulled out as its own pure function so it
    can be tested directly without actually spawning a subprocess."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _start_demo_app() -> tuple[subprocess.Popen, str]:
    """Starts the real demo app as a real subprocess (not in-process, not
    mocked), on an ephemeral port."""
    port = _free_port()
    env = _demo_app_subprocess_env()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cua.demo_app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(REPO_ROOT / "src"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/search", timeout=1)
            break
        except Exception as exc:  # noqa: BLE001 -- readiness poll, retry until deadline
            last_exc = exc
            time.sleep(0.2)
    else:
        proc.terminate()
        raise EvidenceRunFailure(f"demo app never became ready: {last_exc}")
    return proc, base_url


def _stop_demo_app(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _policy() -> PolicyEngine:
    return PolicyEngine(PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                          ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT),
    ))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


async def _run_discovery(
    model: AnthropicModelAdapter, base_url: str, *, discovery_evidence_dir: Path = DISCOVERY_EVIDENCE_DIR
) -> DiscoveryTrace:
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_PLAYWRIGHT_EXECUTABLE_PATH)
    logger = StructuredLogger(run_id="discovery_trace", log_dir=discovery_evidence_dir)
    evidence = DiscoveryEvidenceCapture(discovery_evidence_dir)
    loop = DiscoveryLoop(model, surface, _policy(), logger, evidence, max_steps=15, overall_timeout_s=120)
    try:
        trace = await loop.run(GOAL)
        if trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED:
            # DiscoveryEvidenceCapture only fires on non-GOAL_ACHIEVED stops
            # (a clean success needs no debugging aid) -- take one success
            # screenshot explicitly here so Phase 8's "screenshots where
            # useful" requirement has a real, non-synthetic image to point at.
            discovery_evidence_dir.mkdir(parents=True, exist_ok=True)
            (discovery_evidence_dir / "discovery_success.png").write_bytes(await surface.screenshot())
        return trace
    finally:
        await surface.close()


def _preserve_discovery_failure(
    trace: DiscoveryTrace, model: AnthropicModelAdapter, *, discovery_failure_dir: Path = DISCOVERY_FAILURE_DIR
) -> None:
    """One sanitized failure record -- Phase 8 Section 11: preserve useful
    evidence rather than deleting it, but do not clutter evidence/ with
    many irrelevant attempts. Overwrites any previous failure record from
    an earlier attempt in this same evidence/ directory, so only the most
    recent genuine failure is kept."""
    _write_json(discovery_failure_dir / "discovery_failure_summary.json", {
        "goal": GOAL,
        "stop_reason": trace.stop_reason.value if trace.stop_reason else None,
        "stop_message": trace.stop_message,
        "steps_executed": len(trace.steps),
        "genuine_model_decision_calls": len(model.calls),
        "run_id": trace.run_id,
        "note": "A genuine AnthropicModelAdapter-backed discovery attempt that did not reach GOAL_ACHIEVED. Preserved as-is, not edited to look successful.",
    })


def _build_and_save_artifact(trace: DiscoveryTrace, base_url: str, *, evidence_dir: Path = EVIDENCE_DIR) -> tuple[object, Path]:
    builder = ArtifactBuilder()
    artifact = builder.build(
        trace,
        capability_id=CAPABILITY_ID,
        capability_version=CAPABILITY_VERSION,
        name="Look up member and read savings balance (genuine Phase 8 discovery)",
        description="Built directly from a real AnthropicModelAdapter-backed discovery run against the local synthetic demo app.",
        app_name="CU Ops Console",
        base_url_pattern=base_url,
        inputs=[InputParam(name="member_id", type=ParamType.STRING, description="5-digit synthetic member identifier", example=DISCOVERY_MEMBER_ID)],
        outputs=[OutputParam(name="savings_balance", type=ParamType.FLOAT, description="Current savings balance")],
        parameter_values={"member_id": DISCOVERY_MEMBER_ID},
    )
    fill_steps = [s for s in artifact.steps if s.action == ActionType.FILL]
    if not fill_steps or fill_steps[0].params.get("value") != "{{member_id}}":
        raise EvidenceRunFailure(
            "generated artifact did not parameterize member_id as '{{member_id}}' -- "
            "refusing to persist an artifact that bakes in the literal discovery value."
        )
    saved_path = save_artifact(artifact, evidence_dir)
    # Round-trip validate through the real schema -- Phase 8 Section 6/A.3.
    loaded = load_artifact(saved_path)
    if loaded != artifact:
        raise EvidenceRunFailure(f"artifact failed its own YAML round-trip at {saved_path}")
    return artifact, saved_path


async def _run_replay(artifact, model, *, replay_evidence_dir: Path = REPLAY_EVIDENCE_DIR) -> tuple[object, int, int]:
    calls_before = len(model.calls)
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_PLAYWRIGHT_EXECUTABLE_PATH)
    logger = StructuredLogger(run_id="replay_trace", log_dir=replay_evidence_dir)
    evidence = EvidenceCapture(replay_evidence_dir)
    engine = ReplayEngine(_policy(), logger, evidence)
    try:
        result = await engine.run(artifact, {"member_id": REPLAY_MEMBER_ID}, surface)
        if result.status == RunStatus.SUCCESS:
            replay_evidence_dir.mkdir(parents=True, exist_ok=True)
            (replay_evidence_dir / "replay_success.png").write_bytes(await surface.screenshot())
    finally:
        await surface.close()
    calls_after = len(model.calls)
    return result, calls_before, calls_after


async def _main_async() -> int:
    api_key = _require_api_key()
    model_name = _require_model_name()
    started_at = datetime.now(timezone.utc).isoformat()

    client = AsyncAnthropic(api_key=api_key)
    model = AnthropicModelAdapter(client, model_name)

    print(f"[run_evidence] provider=anthropic model={model_name!r} goal={GOAL!r}")
    demo_proc, base_url = _start_demo_app()
    print(f"[run_evidence] demo app up at {base_url}")

    try:
        trace = await _run_discovery(model, base_url)
        print(f"[run_evidence] discovery stop_reason={trace.stop_reason} genuine_model_decision_calls={len(model.calls)}")

        if trace.stop_reason != DiscoveryStopReason.GOAL_ACHIEVED:
            _preserve_discovery_failure(trace, model)
            print(
                f"[run_evidence] STOP: genuine discovery did not reach GOAL_ACHIEVED "
                f"(stop_reason={trace.stop_reason}, message={trace.stop_message!r}). "
                f"Failure evidence written to {DISCOVERY_FAILURE_DIR}. Not proceeding to artifact/replay.",
                file=sys.stderr,
            )
            return 1

        try:
            artifact, artifact_path = _build_and_save_artifact(trace, base_url)
        except ArtifactBuildError as exc:
            raise EvidenceRunFailure(f"ArtifactBuilder rejected the discovery trace: {exc}") from exc
        print(f"[run_evidence] artifact saved: {artifact_path}")

        calls_after_discovery = len(model.calls)
        result, calls_before_replay, calls_after_replay = await _run_replay(artifact, model)
        print(f"[run_evidence] replay status={result.status} outputs={result.outputs}")
        print(f"[run_evidence] model.calls: after discovery={calls_after_discovery} before replay={calls_before_replay} after replay={calls_after_replay}")

        discovery_summary = {
            "provider": "anthropic",
            "model": model_name,
            "goal": GOAL,
            "discovery_member_id": DISCOVERY_MEMBER_ID,
            "stop_reason": trace.stop_reason.value,
            "run_id": trace.run_id,
            "genuine_model_decision_calls": len(model.calls),
            "steps_executed": len(trace.steps),
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
            "artifact_capability_id": artifact.capability_id,
            "artifact_capability_version": artifact.capability_version,
            "member_id_parameterized_as": "{{member_id}}",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(DISCOVERY_EVIDENCE_DIR / "discovery_summary.json", discovery_summary)

        replay_summary = {
            "replay_member_id": REPLAY_MEMBER_ID,
            "replay_status": result.status.value,
            "replay_code": result.code.value if result.code else None,
            "replay_outputs": result.outputs,
            "expected_savings_balance": EXPECTED_REPLAY_BALANCE,
            "fresh_session": True,
            "model_calls_before_replay": calls_before_replay,
            "model_calls_after_replay": calls_after_replay,
            "model_calls_increased_during_replay": calls_after_replay != calls_before_replay,
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
        }
        _write_json(REPLAY_EVIDENCE_DIR / "replay_summary.json", replay_summary)

        ok = True
        if result.status != RunStatus.SUCCESS:
            print(f"[run_evidence] FAIL: replay status was {result.status}, expected SUCCESS", file=sys.stderr)
            ok = False
        if result.outputs != {"savings_balance": EXPECTED_REPLAY_BALANCE}:
            print(f"[run_evidence] FAIL: replay outputs were {result.outputs}, expected savings_balance={EXPECTED_REPLAY_BALANCE}", file=sys.stderr)
            ok = False
        if calls_after_replay != calls_before_replay:
            print("[run_evidence] FAIL: model.calls increased during replay -- replay must make zero model decisions", file=sys.stderr)
            ok = False

        if not ok:
            return 1

        print("[run_evidence] SUCCESS: genuine discovery -> artifact -> fresh-session replay all verified.")
        return 0
    finally:
        _stop_demo_app(demo_proc)


def main() -> int:
    try:
        return asyncio.run(_main_async())
    except EvidenceRunFailure as exc:
        print(f"[run_evidence] STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
