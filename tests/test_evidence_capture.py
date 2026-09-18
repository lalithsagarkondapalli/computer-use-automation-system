from pathlib import Path

from cua.replay.evidence import EvidenceCapture

from fakes import FakeSurfaceAdapter


async def test_capture_writes_screenshot_file(tmp_path: Path):
    capture = EvidenceCapture(tmp_path)
    ref = await capture.capture("run-123", FakeSurfaceAdapter())
    assert ref is not None
    assert ref.kind == "screenshot"
    assert Path(ref.path).read_bytes() == b"fake-png-bytes"


async def test_capture_returns_none_when_screenshot_fails(tmp_path: Path):
    class BrokenSurface(FakeSurfaceAdapter):
        async def screenshot(self) -> bytes:
            raise RuntimeError("browser crashed")

    capture = EvidenceCapture(tmp_path)
    assert await capture.capture("run-456", BrokenSurface()) is None
