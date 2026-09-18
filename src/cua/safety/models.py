"""Policy domain models: the allowlist configuration and the typed
decision a PolicyEngine (implemented in Phase 3) returns.

Deliberately split from the enforcing engine: the *shape* of a policy
(what can be configured) is a schema decision worth reviewing on its own,
independent of the code that reads it. `PolicyConfig` is the "explicit,
configurable allowlist" Section 3.4 requires; `PolicyDecision` is what lets
a caller act on a policy check without inspecting exception types.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from cua.actions import ActionType
from cua.artifact.schema import RiskLevel


class PolicyConfig(BaseModel):
    allowed_domains: tuple[str, ...]
    allowed_routes: tuple[str, ...] = Field(default_factory=tuple)  # empty = all routes on an allowed domain
    allowed_actions: tuple[ActionType, ...]
    risk_requires_approval: tuple[RiskLevel, ...] = (RiskLevel.SENSITIVE, RiskLevel.IRREVERSIBLE)
    risk_blocked: tuple[RiskLevel, ...] = ()  # e.g. (IRREVERSIBLE,) to hard-block regardless of approval


class PolicyDecision(BaseModel):
    allowed: bool
    requires_approval: bool = False
    reason: str
