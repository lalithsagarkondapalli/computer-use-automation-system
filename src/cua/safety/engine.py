"""PolicyEngine: enforces the allowlist configured in PolicyConfig.

Every check here runs BEFORE the corresponding action reaches a
SurfaceAdapter (Section 3.4). `ReplayEngine` calls `check_domain` before
every NAVIGATE and again after every NAVIGATE/CLICK (to catch a redirect
that escapes the allowlist -- see Section 3.4's specific warning about
navigation), and calls `check_action` before every step regardless of
action type.
"""
from __future__ import annotations

from urllib.parse import urlparse

from cua.artifact.schema import Step
from cua.safety.models import PolicyConfig, PolicyDecision

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self._config = config

    def check_domain(self, url: str) -> PolicyDecision:
        """Reject anything not http(s), and anything not on the allowed
        domain (and, if configured, route) list. Used both for the
        pre-navigation URL and for the post-action URL, so a redirect that
        lands somewhere off-allowlist is caught even though the original
        navigation target was fine.
        """
        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme not in _ALLOWED_SCHEMES:
            return PolicyDecision(allowed=False, reason=f"scheme {parsed.scheme!r} is not permitted")
        host = parsed.hostname or ""
        if not any(host == d or host.endswith(f".{d}") for d in self._config.allowed_domains):
            return PolicyDecision(
                allowed=False,
                reason=f"host {host!r} is not in the allowed domain list {self._config.allowed_domains}",
            )
        if self._config.allowed_routes and not any(parsed.path.startswith(r) for r in self._config.allowed_routes):
            return PolicyDecision(allowed=False, reason=f"route {parsed.path!r} is not in the allowed route list")
        return PolicyDecision(allowed=True, reason="domain and route allowed")

    def check_action(self, step: Step) -> PolicyDecision:
        """Reject an action type the policy doesn't allow, hard-block a
        risk level the policy marks unconditionally blocked, and require
        approval (currently always unmet -- see Phase 5) for any other
        configured risky risk level.
        """
        if step.action not in self._config.allowed_actions:
            return PolicyDecision(allowed=False, reason=f"action {step.action.value!r} is not in the allowed action list")
        if step.risk in self._config.risk_blocked:
            return PolicyDecision(allowed=False, reason=f"risk level {step.risk.value} is hard-blocked by policy")
        if step.risk in self._config.risk_requires_approval:
            return PolicyDecision(
                allowed=False,
                requires_approval=True,
                reason=(
                    f"risk level {step.risk.value} requires human approval before executing "
                    "(no approval path exists yet in this build -- see Phase 5 handoff)"
                ),
            )
        return PolicyDecision(allowed=True, reason="action and risk level allowed")
