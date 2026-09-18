"""Phase 6 end-to-end tests: the real demo app (uvicorn subprocess, real
FastAPI/Jinja2 rendering), a real headless Chromium session, and the real
ReplayEngine -- no FakeSurfaceAdapter anywhere in this file. Loads the
actual persisted artifacts/member_lookup_balance@1.0.0.yaml through the
real `cua.artifact.store.load_artifact`, so these tests prove the saved
artifact itself is executable, not a hand-rolled equivalent.

The three synthetic member IDs are never special-cased in this test file's
own assertions about *how* the outcome is reached -- SUCCESS/BUSINESS_OUTCOME/
ESCALATION_REQUIRED all come from the artifact's checkpoint + probes being
evaluated against whatever the demo app's real HTML actually renders for
that ID (see cua.demo_app.data.lookup and templates/results.html).
"""
from __future__ import annotations

import os
from pathlib import Path

from cua.artifact.store import load_artifact
from cua.handoff import ControlState, HandoffController
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.playwright_web import PlaywrightWebSurface

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "member_lookup_balance@1.0.0.yaml"
_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")

# demo_app_base_url fixture now lives in tests/conftest.py (Phase 7 --
# shared with test_full_contract_e2e.py rather than duplicated).


def _artifact_for(base_url: str):
    artifact = load_artifact(ARTIFACT_PATH)
    return artifact.model_copy(update={"target": artifact.target.model_copy(update={"base_url_pattern": base_url})})


def _engine(tmp_path: Path, handoff: HandoffController | None = None) -> ReplayEngine:
    policy = PolicyEngine(PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=("navigate", "click", "fill", "select", "read_extract", "checkpoint", "wait"),
    ))
    logger = StructuredLogger(run_id="demo-e2e", log_dir=tmp_path / "logs")
    evidence = EvidenceCapture(tmp_path / "evidence")
    return ReplayEngine(policy, logger, evidence, handoff)


async def test_success_real_browser_real_demo_app(tmp_path: Path, demo_app_base_url: str):
    artifact = _artifact_for(demo_app_base_url)
    engine = _engine(tmp_path)
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    try:
        result = await engine.run(artifact, {"member_id": "12345"}, surface)
    finally:
        await surface.close()

    assert result.status == RunStatus.SUCCESS
    assert result.code is None
    assert result.outputs == {"savings_balance": 1234.56}


async def test_business_outcome_customer_not_found_real_browser(tmp_path: Path, demo_app_base_url: str):
    artifact = _artifact_for(demo_app_base_url)
    engine = _engine(tmp_path)
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    try:
        result = await engine.run(artifact, {"member_id": "00000"}, surface)
    finally:
        await surface.close()

    assert result.status == RunStatus.BUSINESS_OUTCOME
    assert result.code == OutcomeCode.CUSTOMER_NOT_FOUND
    assert result.outputs is None


async def test_compliance_hold_escalates_human_clears_it_on_same_session_then_succeeds(tmp_path: Path, demo_app_base_url: str):
    """The full Section-10 recipe against the real demo app: automation
    hits the real Compliance Hold interstitial, escalates, a "human"
    (test code) clears it on the SAME live page via the SurfaceAdapter's
    own public methods, resumes, and the deterministic checkpoint re-check
    lets replay continue to a real SUCCESS."""
    artifact = _artifact_for(demo_app_base_url)
    handoff = HandoffController()
    engine = _engine(tmp_path, handoff)
    surface = await PlaywrightWebSurface.launch(headed=False, executable_path=_EXECUTABLE_PATH)
    member_id_cell = TargetDescriptor(dom_selector="#member-results tr:nth-of-type(1) td:nth-of-type(2)")

    try:
        import asyncio

        run_task = asyncio.ensure_future(engine.run(artifact, {"member_id": "55555"}, surface))

        for _ in range(300):
            if handoff.state == ControlState.HUMAN_CONTROL:
                break
            await asyncio.sleep(0.02)
        assert handoff.state == ControlState.HUMAN_CONTROL
        assert not run_task.done()

        # -- session-continuity proof, beyond Python object identity --
        url_before = await surface.current_url()
        assert "member_id=55555" in url_before
        # the member-id cell already exists in the SAME live DOM, just
        # hidden behind the modal -- proves the record was rendered once,
        # server-side, not re-fetched after the human acts
        resolved_cell = await surface.resolve(member_id_cell)
        member_id_text_before = await surface.read(resolved_cell, "text")
        assert member_id_text_before == "55555"

        record = handoff.current_intervention
        assert record is not None
        assert record.request.reason_code == OutcomeCode.UNKNOWN_DIALOG.value
        assert "member_id=55555" in record.request.current_url

        # -- the human acts on the SAME live browser page, through the
        # SurfaceAdapter's existing public methods only --
        clear_hold = await surface.resolve(TargetDescriptor(role="button", accessible_name="Clear Hold"))
        await surface.click(clear_hold)

        # continuity again, post-action: same page (no reload/navigation --
        # "Clear Hold" only sets location.hash, a same-document URL change,
        # never a real navigation), same DOM node still readable with the
        # same content
        url_after_click = await surface.current_url()
        assert url_after_click.split("#")[0] == url_before.split("#")[0]  # no navigation ever happened
        assert url_after_click == url_before + "#hold-cleared"  # the expected same-document hash update
        resolved_cell_after = await surface.resolve(member_id_cell)
        member_id_text_after = await surface.read(resolved_cell_after, "text")
        assert member_id_text_after == member_id_text_before == "55555"

        handoff.request_resume()
        result = await asyncio.wait_for(run_task, timeout=15)
    finally:
        await surface.close()

    assert result.status == RunStatus.SUCCESS
    assert result.outputs == {"savings_balance": 9876.54}
    assert handoff.state == ControlState.AUTOMATION_CONTROL
    assert len(handoff.interventions) == 1
    assert handoff.interventions[0].resolved is True
