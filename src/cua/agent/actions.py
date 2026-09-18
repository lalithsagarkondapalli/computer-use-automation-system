"""AgentAction: a single proposed action from the discovery loop's model.

This is the *only* channel through which the LLM affects the system: the
model returns exactly one of these -- via structured tool-calling, see the
Phase 4 `AnthropicModelAdapter` -- never free text or code. Every field the
policy engine needs to make an allow/block decision (Section 3.4) is
present here, before any surface interaction happens.

`reasoning` is required and is *not* optional decoration: it is what makes
the discovery evidence log auditable ("what did the model think it was
doing, and why") without persisting the raw model transcript. It is
discovery-only; nothing in `cua.artifact` or `cua.replay` has a `reasoning`
field, since replay makes no decisions to explain.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from cua.actions import ActionType
from cua.artifact.targets import TargetDescriptor


class AgentAction(BaseModel):
    action: ActionType
    target: TargetDescriptor | None = None
    params: dict[str, str] = Field(default_factory=dict)
    reasoning: str

    @model_validator(mode="after")
    def _action_shape(self) -> "AgentAction":
        needs_target = {ActionType.CLICK, ActionType.FILL, ActionType.SELECT}
        if self.action in needs_target and self.target is None:
            raise ValueError(f"{self.action} requires a target")
        if self.action == ActionType.FILL and "value" not in self.params:
            raise ValueError("FILL requires params['value']")
        if self.action == ActionType.SELECT and "value" not in self.params:
            raise ValueError("SELECT requires params['value']")
        if self.action == ActionType.NAVIGATE and "route" not in self.params and "url" not in self.params:
            raise ValueError("NAVIGATE requires params['route'] or params['url']")
        if self.action == ActionType.REQUEST_HUMAN and "reason" not in self.params:
            raise ValueError("REQUEST_HUMAN requires params['reason']")
        return self
