from cua.actions import ActionType
from cua.artifact.schema import RiskLevel, Step
from cua.artifact.targets import TargetDescriptor
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig


def _engine(**overrides) -> PolicyEngine:
    defaults = dict(
        allowed_domains=("127.0.0.1", "localhost"),
        allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.READ_EXTRACT, ActionType.CHECKPOINT),
    )
    defaults.update(overrides)
    return PolicyEngine(PolicyConfig(**defaults))


def test_allowed_domain_succeeds():
    assert _engine().check_domain("http://127.0.0.1:8000/search").allowed is True


def test_unapproved_domain_blocked():
    assert _engine().check_domain("http://evil.example.com/steal").allowed is False


def test_disallowed_scheme_blocked():
    assert _engine().check_domain("javascript:alert(1)").allowed is False


def test_subdomain_of_allowed_domain_is_allowed():
    assert _engine(allowed_domains=("example.com",)).check_domain("https://app.example.com/x").allowed is True


def test_unrelated_domain_containing_allowed_domain_as_substring_is_blocked():
    # "notexample.com" must not be treated as a match for allowed domain "example.com".
    assert _engine(allowed_domains=("example.com",)).check_domain("https://notexample.com/x").allowed is False


def test_route_allowlist_enforced_when_configured():
    engine = _engine(allowed_routes=("/search", "/results"))
    assert engine.check_domain("http://127.0.0.1:8000/search").allowed is True
    assert engine.check_domain("http://127.0.0.1:8000/admin").allowed is False


def test_allowed_action_succeeds():
    engine = _engine()
    step = Step(step_id="s", action=ActionType.CLICK, target=TargetDescriptor(dom_selector="#x"))
    assert engine.check_action(step).allowed is True


def test_prohibited_action_type_blocked():
    engine = _engine(allowed_actions=(ActionType.NAVIGATE,))
    step = Step(step_id="s", action=ActionType.CLICK, target=TargetDescriptor(dom_selector="#x"))
    assert engine.check_action(step).allowed is False


def test_irreversible_risk_requires_approval_by_default():
    engine = _engine()
    step = Step(step_id="s", action=ActionType.CLICK, target=TargetDescriptor(dom_selector="#x"), risk=RiskLevel.IRREVERSIBLE)
    decision = engine.check_action(step)
    assert decision.allowed is False
    assert decision.requires_approval is True


def test_hard_blocked_risk_is_never_approvable():
    engine = _engine(risk_blocked=(RiskLevel.IRREVERSIBLE,))
    step = Step(step_id="s", action=ActionType.CLICK, target=TargetDescriptor(dom_selector="#x"), risk=RiskLevel.IRREVERSIBLE)
    decision = engine.check_action(step)
    assert decision.allowed is False
    assert decision.requires_approval is False


def test_safe_action_does_not_require_approval():
    engine = _engine()
    step = Step(step_id="s", action=ActionType.CLICK, target=TargetDescriptor(dom_selector="#x"), risk=RiskLevel.SAFE)
    decision = engine.check_action(step)
    assert decision.allowed is True
    assert decision.requires_approval is False
