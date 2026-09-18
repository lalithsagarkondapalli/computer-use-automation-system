"""OperatorConsole tests: a real stdlib HTTP server on a background
thread (Section 4), hit with real HTTP requests (urllib), not mocked.
Proves the rendered page shows the intervention's context and that POSTing
/resume actually flips the real HandoffController's state -- exercising
the one thread-boundary crossing (`call_soon_threadsafe`) for real.
"""
from __future__ import annotations

import asyncio
import urllib.request
from pathlib import Path

from cua.handoff import ControlState, HandoffController, InterventionRequest
from cua.handoff.console import OperatorConsole
from cua.results.taxonomy import EvidenceRef


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read()


def _post(url: str) -> int:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


async def test_index_shows_no_open_intervention_before_any_escalation():
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    try:
        status, body = await asyncio.to_thread(_get, console.url)
        assert status == 200
        assert b"No open intervention" in body
        assert b"AUTOMATION_CONTROL" in body
    finally:
        console.stop()


async def test_index_renders_intervention_context_and_a_resume_button(tmp_path: Path):
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    try:
        request = InterventionRequest.create(
            run_id="run-99", mode="replay", reason_code="AMBIGUOUS_CONTROL",
            reason_message="two candidates matched", current_url="http://127.0.0.1:8000/search",
            capability_id="member_lookup_balance", capability_version="1.0.0",
            step_id="s3_click", expected_state="exactly one control matched",
            observed_state="2 candidates matched via role_name",
        )
        await controller.escalate(request)

        status, body = await asyncio.to_thread(_get, console.url)
        text = body.decode("utf-8")

        assert status == 200
        assert "HUMAN_CONTROL" in text
        assert "AMBIGUOUS_CONTROL" in text
        assert "two candidates matched" in text
        assert "member_lookup_balance" in text
        assert "http://127.0.0.1:8000/search" in text
        assert '<form method="POST" action="/resume">' in text
        assert "(no screenshot captured)" in text  # no evidence attached to this request
    finally:
        console.stop()


async def test_index_links_a_screenshot_when_evidence_is_attached(tmp_path: Path):
    png_path = tmp_path / "evidence.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    try:
        request = InterventionRequest.create(
            run_id="run-1", mode="discovery", reason_code="STUCK_STATE", reason_message="repeated the same click",
            current_url="http://127.0.0.1:8000/", evidence=EvidenceRef(kind="screenshot", path=str(png_path)),
        )
        await controller.escalate(request)

        _, body = await asyncio.to_thread(_get, console.url)
        assert b'<img src="/screenshot"' in body

        status, screenshot_bytes = await asyncio.to_thread(_get, console.url + "screenshot")
        assert status == 200
        assert screenshot_bytes == png_path.read_bytes()
    finally:
        console.stop()


async def test_missing_screenshot_file_is_a_clean_404_not_a_crash():
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    try:
        request = InterventionRequest.create(
            run_id="run-1", mode="discovery", reason_code="STUCK_STATE", reason_message="x",
            current_url="http://127.0.0.1:8000/", evidence=EvidenceRef(kind="screenshot", path="/no/such/file.png"),
        )
        await controller.escalate(request)

        import urllib.error
        try:
            await asyncio.to_thread(_get, console.url + "screenshot")
            assert False, "expected a 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        console.stop()


async def test_posting_resume_actually_transitions_the_real_controller():
    """The one real thread-safety concern this module has: the HTTP
    handler runs on a background thread, and `HandoffController` (and its
    underlying `asyncio.Event`s) belongs to the event loop that started
    it. This proves `call_soon_threadsafe` really does bridge that -- a
    plain call from the wrong thread would silently do nothing or crash."""
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    try:
        await controller.escalate(InterventionRequest.create(
            run_id="run-1", mode="discovery", reason_code="HUMAN_REQUESTED", reason_message="x",
            current_url="http://127.0.0.1:8000/",
        ))
        assert controller.state == ControlState.HUMAN_CONTROL

        status = await asyncio.to_thread(_post, console.url + "resume")
        assert status in (200, 303)

        # the console schedules request_resume() onto the event loop via
        # call_soon_threadsafe rather than calling it inline -- give the
        # loop one tick to actually run that callback before asserting.
        for _ in range(50):
            if controller.state == ControlState.RESUME_REQUESTED:
                break
            await asyncio.sleep(0.01)
        assert controller.state == ControlState.RESUME_REQUESTED
    finally:
        console.stop()


async def test_stop_actually_shuts_down_the_server():
    controller = HandoffController()
    console = OperatorConsole(controller, port=0)
    console.start()
    url = console.url
    console.stop()

    import urllib.error
    try:
        await asyncio.to_thread(_get, url)
        assert False, "expected the connection to fail once the server is stopped"
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
