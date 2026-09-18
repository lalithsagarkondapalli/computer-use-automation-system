"""Failure evidence capture: at least one richer debugging signal per
failing run (Section 3.5), captured through the SurfaceAdapter so it works
the same way regardless of which concrete surface is in use.
"""
from __future__ import annotations

from pathlib import Path

from cua.results.taxonomy import EvidenceRef
from cua.surfaces.base import SurfaceAdapter


class EvidenceCapture:
    def __init__(self, directory: Path):
        self._directory = directory

    async def capture(self, run_id: str, surface: SurfaceAdapter) -> EvidenceRef | None:
        """Best-effort: a failure while capturing evidence for a failure
        must never itself crash the run and hide the original error, so
        this deliberately catches broadly and returns None rather than
        propagating. (This is the one place in the codebase that catches
        bare Exception -- see REPORT.md for why it's reviewed and kept
        narrow to this single call.)
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{run_id}.png"
        try:
            data = await surface.screenshot()
        except Exception:
            return None
        path.write_bytes(data)
        return EvidenceRef(kind="screenshot", path=str(path))
