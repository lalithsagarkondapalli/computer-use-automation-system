"""Discovery-side evidence capture: a best-effort screenshot on every
discovery stop condition (Section 5's observability requirement, and the
general "capture evidence of what happened" principle Section 3.5 already
established for replay via `cua.replay.evidence.EvidenceCapture`).

This is a deliberate, small duplication of that class rather than a
shared import: `cua.replay` must never be imported from `cua.agent` (see
tests/test_discovery_replay_isolation.py), and the alternative -- moving
the shared logic somewhere both packages import from -- is a real option
but not one to make silently inside a phase that was told not to
redesign Phase 3 without cause. The two implementations are ~10 lines
each and independently trivial to verify; see the Phase 3 review notes
for the same "duplication over a forced shared abstraction" call made for
FakeSurfaceAdapter/PlaywrightWebSurface's locator-resolution logic.
"""
from __future__ import annotations

from pathlib import Path

from cua.results.taxonomy import EvidenceRef
from cua.surfaces.base import SurfaceAdapter


class DiscoveryEvidenceCapture:
    def __init__(self, directory: Path):
        self._directory = directory

    async def capture(self, run_id: str, surface: SurfaceAdapter) -> EvidenceRef | None:
        """Best-effort -- a failure capturing evidence must never crash
        discovery or hide the real stop reason, so this deliberately
        catches broadly and returns None rather than propagating."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{run_id}.png"
        try:
            data = await surface.screenshot()
        except Exception:
            return None
        path.write_bytes(data)
        return EvidenceRef(kind="screenshot", path=str(path))
