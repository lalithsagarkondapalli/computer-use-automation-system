"""Observation: the bounded slice of UI state given to the model each
decision step.

Section 6 requires that "the model receives only information necessary to
make the next decision." This model is that contract: a summarized,
role-based view of interactive elements plus a screenshot reference, not
the raw page HTML. That keeps token cost sane and mirrors how a human
operator actually perceives a screen -- by its interactive controls, not
its markup -- which is also what makes the same shape plausible for a
screenshot/accessibility-tree-driven desktop surface later (see REPORT.md
Section 4).
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
