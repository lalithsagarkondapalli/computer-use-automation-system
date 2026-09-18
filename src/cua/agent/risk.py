"""Deterministic risk classification for a live discovery proposal.

`AgentAction` deliberately has no `risk` field the model can set itself --
Section 3.4's guardrails are meaningless if the actor being guarded
against also gets to self-report how dangerous its own proposal is. This
module is the system-side, keyword-based classifier that stands in for
that judgment instead: conservative, auditable by reading this file, and
independent of anything the model says in `reasoning`.

It deliberately errs toward over-classifying as risky rather than under --
a false positive here costs a policy block (and, if the policy path
existed, a human approval); a false negative could let an irreversible
action through unexamined. This is the same reason `RiskLevel.IRREVERSIBLE`
is hard-blocked by default in `PolicyConfig` (see `cua.safety.models`).

This is intentionally a small, reviewable lookup, not a model call: asking
an LLM "is this risky?" would just move the same trust problem one layer
deeper.
"""
from __future__ import annotations

from cua.agent.actions import AgentAction
from cua.artifact.schema import RiskLevel

# Checked in this order; the first matching level wins. Kept short and
# literal on purpose -- broaden this list under review, not by guessing
# what an LLM might phrase differently.
_IRREVERSIBLE_KEYWORDS: tuple[str, ...] = (
    "delete", "close account", "remove", "cancel", "terminate",
    "unsubscribe", "deactivate", "permanently",
)
_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "transfer", "submit", "confirm", "withdraw", "pay", "send money",
    "approve", "authorize", "wire",
)


def classify_action_risk(action: AgentAction) -> RiskLevel:
    """Inspect only structural, system-visible fields of the proposal --
    the target's own signals and the action's params -- never `reasoning`
    (the model's own narration is not evidence of what the action
    actually does)."""
    haystack = " ".join(
        filter(
            None,
            [
                action.target.accessible_name if action.target else None,
                action.target.visible_text if action.target else None,
                action.target.nearby_anchor if action.target else None,
                action.params.get("value"),
                action.params.get("route"),
                action.params.get("url"),
            ],
        )
    ).lower()
    if any(keyword in haystack for keyword in _IRREVERSIBLE_KEYWORDS):
        return RiskLevel.IRREVERSIBLE
    if any(keyword in haystack for keyword in _SENSITIVE_KEYWORDS):
        return RiskLevel.SENSITIVE
    return RiskLevel.SAFE
