"""Observation: the bounded slice of UI state a SurfaceAdapter can report
about its current screen.

Moved here from `cua.agent` during Phase 3: this is surface-perception
data (what any SurfaceAdapter can tell a caller about the current state),
not an agent-decision concept -- the discovery loop consumes it, but the
type itself has nothing to do with an LLM. Keeping it under `cua.surfaces`
means `SurfaceAdapter` (which every replay path also depends on, via
`cua.replay`) never has to import anything from `cua.agent` or
`cua.models`. See REPORT.md / the Phase 3 review notes for why this moved.

Section 6 requires that "the model receives only information necessary to
make the next decision" -- Observation is that contract: a summarized,
role-based view of interactive elements plus a screenshot reference, not
raw page HTML. That keeps token cost sane and mirrors how a human
operator actually perceives a screen -- by its interactive controls, not
its markup -- which is also what makes the same shape plausible for a
screenshot/accessibility-tree-driven desktop surface later.
"""
from __future__ import annotations

from pydantic import BaseModel


class InteractiveElement(BaseModel):
    """One control the model can see and potentially act on."""

    role: str
    accessible_name: str | None = None
    text: str | None = None
    control_type: str | None = None


class Observation(BaseModel):
    url: str
    title: str
    interactive_elements: list[InteractiveElement]
    visible_text_summary: str
    screenshot_path: str | None = None
    step_number: int
