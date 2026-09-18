"""VALIDATE: the discovery loop's own semantic sanity check on a proposal,
distinct from (and after) the schema validation `AgentAction` already
enforces at construction time.

Pydantic can confirm a proposal is *shaped* correctly (a CLICK has a
target, a FILL has a value, ...) but it cannot confirm the proposal is
*grounded* in reality -- that the control the model wants to click was
actually part of what it was just shown, rather than invented. Section
6's "the model proposes, the system validates" is only meaningful if that
validation includes checking the proposal against the observation it was
supposedly a response to; without this, a model that hallucinates a
plausible-sounding control would sail straight through to POLICY CHECK
and ACT, where it would simply fail as TargetNotFound -- correct, but
late, and indistinguishable in the log from a genuinely-flaky UI.

Returns a short human-readable rejection reason, or None if the proposal
passes. Never raises -- an invalid proposal is an ordinary, expected
outcome of asking a model to decide, not an exceptional one.
"""
from __future__ import annotations

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.surfaces.observation import Observation

_GROUNDING_REQUIRED = frozenset({ActionType.CLICK, ActionType.FILL, ActionType.SELECT, ActionType.WAIT, ActionType.READ_EXTRACT})


def validate_proposal(action: AgentAction, observation: Observation) -> str | None:
    if action.action in _GROUNDING_REQUIRED:
        reason = _grounding_error(action, observation)
        if reason is not None:
            return reason

    if action.action == ActionType.READ_EXTRACT:
        output_name = action.params.get("output_name", "")
        if not output_name or not output_name.isidentifier():
            return f"READ_EXTRACT requires a valid params['output_name'] identifier, got {output_name!r}"

    if action.action == ActionType.CHECKPOINT:
        description = action.params.get("description", "").strip()
        if not description:
            return "CHECKPOINT requires a non-empty params['description']"

    return None


def _grounding_error(action: AgentAction, observation: Observation) -> str | None:
    target = action.target
    if target is None:
        return f"{action.action.value} requires a target"

    if target.role and target.accessible_name:
        if any(
            e.role == target.role and e.accessible_name == target.accessible_name
            for e in observation.interactive_elements
        ):
            return None

    if target.nearby_anchor and target.nearby_anchor.lower() in observation.visible_text_summary.lower():
        return None

    if target.visible_text and target.visible_text.lower() in observation.visible_text_summary.lower():
        return None

    if target.dom_selector:
        # dom_selector is a discovery-time fallback the loop cannot verify
        # against a summarized Observation (Observation has no raw DOM) --
        # let it through to POLICY CHECK / ACT, where a real TargetNotFound
        # will surface if it doesn't exist. This is the one signal grounding
        # cannot check here, by construction (see TargetDescriptor's
        # docstring: dom_selector is last-resort, discovery-time scaffolding).
        return None

    return (
        f"target for {action.action.value} (role={target.role!r}, accessible_name={target.accessible_name!r}, "
        f"nearby_anchor={target.nearby_anchor!r}, visible_text={target.visible_text!r}) "
        "is not grounded in anything the current observation actually shows"
    )
