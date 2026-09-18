from cua.actions import ActionType
from cua.artifact.schema import RiskLevel
from cua.safety.models import PolicyConfig, PolicyDecision


def test_policy_config_defaults():
    cfg = PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(ActionType.NAVIGATE, ActionType.CLICK, ActionType.READ_EXTRACT),
    )
    assert RiskLevel.IRREVERSIBLE in cfg.risk_requires_approval
    assert RiskLevel.SENSITIVE in cfg.risk_requires_approval
    assert cfg.risk_blocked == ()
    assert cfg.allowed_routes == ()


def test_policy_config_can_hard_block_irreversible_actions():
    cfg = PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(ActionType.CLICK,),
        risk_blocked=(RiskLevel.IRREVERSIBLE,),
    )
    assert RiskLevel.IRREVERSIBLE in cfg.risk_blocked


def test_policy_decision_shape():
    decision = PolicyDecision(allowed=False, requires_approval=True, reason="risky action needs approval")
    assert decision.allowed is False
    assert decision.requires_approval is True
