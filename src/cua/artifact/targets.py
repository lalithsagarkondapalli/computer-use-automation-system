"""TargetDescriptor: a multi-signal, surface-agnostic way to identify a
control on screen.

Section 8 of the assignment is explicit that CSS/XPath selectors must not
be the architectural *foundation* for targeting: a selector that works
today breaks the moment a legacy app's markup shifts even slightly, and it
tells a human reviewer nothing about *why* a control is "the search
button." A TargetDescriptor instead records every signal discovery
observed about a control, in decreasing order of how well that signal
survives markup churn (and even survives a port to a desktop surface,
where accessibility trees still exist but DOM selectors do not).
`dom_selector` is present only as the last-resort fallback, never the
primary signal a resolver reaches for first.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# Tried in this order by SurfaceAdapter.resolve(); the first strategy that
# yields exactly one match wins. role_name, label, and text survive
# legacy/no-DOM surfaces; dom_selector is discovery-time-only scaffolding.
DEFAULT_STRATEGY_ORDER: tuple[str, ...] = ("role_name", "label", "text", "dom_selector")


class TargetDescriptor(BaseModel):
    role: str | None = None
    accessible_name: str | None = None
    visible_text: str | None = None
    semantic_relationship: str | None = None  # e.g. "row containing the searched member"
    nearby_anchor: str | None = None  # e.g. "label reading 'Member ID'"
    dom_selector: str | None = None  # last-resort fallback recorded at discovery time
    frame_path: tuple[str, ...] = Field(default_factory=tuple)  # iframe traversal, outermost first
    expected_control_type: str | None = None  # e.g. "input[type=text]" -- a sanity check, not a locator
    strategy_order: tuple[str, ...] = DEFAULT_STRATEGY_ORDER

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> "TargetDescriptor":
        # nearby_anchor added to this check in Phase 3: once "label" became a
        # real resolution strategy (get_by_label), a descriptor carrying only
        # a label anchor is a legitimate, sufficient target -- the original
        # Phase 2 check predated that strategy and would reject it.
        if not any([self.role, self.accessible_name, self.visible_text, self.nearby_anchor, self.dom_selector]):
            raise ValueError(
                "TargetDescriptor needs at least one identifying signal "
                "(role, accessible_name, visible_text, nearby_anchor, or dom_selector)"
            )
        if self.role is not None and self.accessible_name is None:
            raise ValueError(
                "role must be paired with accessible_name -- an unnamed role "
                "(e.g. 'button') is not enough to resolve a unique control"
            )
        return self
