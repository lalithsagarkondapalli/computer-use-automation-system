"""ReplayEngine integration-style unit tests, all against FakeSurfaceAdapter
(no browser needed -- Playwright itself is exercised separately in
test_playwright_web_surface.py). These are the tests that prove the six
outcome categories are behaviorally distinct, not collapsed into generic
exceptions, and that policy runs before the surface is ever touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cua.actions import ActionType
from cua.artifact.checks import CheckpointSpec, ConditionProbe, TextPresentCheck, UrlMatchesCheck
from cua.artifact.extraction import ExtractionSpec
from cua.artifact.schema import CapabilityArtifact, InputParam, ParamType, RetryPolicy, Step
from cua.artifact.targets import TargetDescriptor
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig

from factories import base_kwargs
from fakes import FakeElement, FakeSurfaceAdapter


def _member_lookup_steps() -> list[Step]:
    return [
        Step(step_id="s1_navigate", action=ActionType.NAVIGATE, params={"route": "/search"}),
        Step(
            step_id="s2_fill",
            action=ActionType.FILL,
            target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
            params={"value": "{{member_id}}"},
        ),
        Step(
            step_id="s3_click",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Search"),
        ),
        Step(
            step_id="s4_checkpoint",
            action=ActionType.CHECKPOINT,
            checkpoint=CheckpointSpec(
                description="results page reached",
                assertion=UrlMatchesCheck(pattern="/results"),
                probes=[
                    ConditionProbe(
                        description="not-found message shown",
                        assertion=TextPresentCheck(text="No member found"),
                        status=RunStatus.BUSINESS_OUTCOME,
                        code=OutcomeCode.CUSTOMER_NOT_FOUND,
                    )
                ],
            ),
        ),
        Step(
            step_id="s5_extract",
            action=ActionType.READ_EXTRACT,
            target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
            extraction=ExtractionSpec(
                output_name="savings_balance",
                target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
                transform="currency_to_float",
            ),
        ),
    ]


def _member_lookup_artifact(**overrides) -> CapabilityArtifact:
    kwargs = base_kwargs(_member_lookup_steps())
    kwargs.update(overrides)
    return CapabilityArtifact(**kwargs)


def _minimal_artifact(steps: list[Step]) -> CapabilityArtifact:
    kwargs = base_kwargs(steps)
    kwargs["inputs"] = []
    return CapabilityArtifact(**kwargs)


def _make_engine(tmp_path: Path, **policy_overrides) -> ReplayEngine:
    defaults = dict(
        allowed_domains=("127.0.0.1",),
        allowed_actions=(
            ActionType.NAVIGATE,
            ActionType.CLICK,
            ActionType.FILL,
            ActionType.SELECT,
            ActionType.READ_EXTRACT,
            ActionType.CHECKPOINT,
            ActionType.WAIT,
        ),
    )
    defaults.update(policy_overrides)
    policy = PolicyEngine(PolicyConfig(**defaults))
    logger = StructuredLogger(run_id="test-run", log_dir=tmp_path / "logs")
    evidence = EvidenceCapture(tmp_path / "evidence")
    return ReplayEngine(policy, logger, evidence)


# Calls that represent the surface actually being *acted on* -- as
# opposed to `screenshot`, which is observability/evidence capture and is
# legitimately allowed even on a run that never executes a single business
# action (Section 3.5 requires failure evidence; it must not be confused
# with "the surface was touched").
_BUSINESS_ACTIONS = frozenset({"navigate", "click", "fill", "select", "read"})


def _business_action_calls(surface: FakeSurfaceAdapter) -> list[tuple]:
    return [call for call in surface.calls if call[0] in _BUSINESS_ACTIONS]


def _register_happy_path(surface: FakeSurfaceAdapter, *, balance: str = "$1,234.56", not_found: bool = False) -> None:
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.script_checkpoint("No member found", not_found)
    surface.script_checkpoint("/results", True)
    if not not_found:
        surface.register_role_name("status", "Savings Balance", [FakeElement(text=balance)])


# -- SUCCESS -----------------------------------------------------------------

async def test_success_returns_typed_output_with_zero_recovery_attempts(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    _register_happy_path(surface)
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.SUCCESS
    assert result.code is None
    assert result.outputs == {"savings_balance": 1234.56}
    assert result.recovery_attempts == []
    assert result.capability_id == "member_lookup_balance"


# -- BUSINESS_OUTCOME ----------------------------------------------------------

async def test_business_outcome_customer_not_found_is_not_a_crash(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    _register_happy_path(surface, not_found=True)
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "00000"}, surface)

    assert result.status == RunStatus.BUSINESS_OUTCOME
    assert result.code == OutcomeCode.CUSTOMER_NOT_FOUND
    assert result.outputs is None
    assert result.failing_step_id == "s4_checkpoint"
    # a business outcome still captures evidence -- it's a meaningful stop, just not a bug
    assert len(result.evidence) == 1


# -- ESCALATION_REQUIRED ---------------------------------------------------

async def test_ambiguous_control_escalates_rather_than_guessing(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])  # two candidates, no disambiguator
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.ESCALATION_REQUIRED
    assert result.code == OutcomeCode.AMBIGUOUS_CONTROL
    assert result.failing_step_id == "s3_click"
    assert "2 candidates" in result.observed_state


# -- HARD_FAILURE ------------------------------------------------------------

async def test_checkpoint_never_satisfied_is_hard_failure(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.script_checkpoint("No member found", False)
    surface.script_checkpoint("/results", False)  # never reached
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.HARD_FAILURE
    assert result.code == OutcomeCode.CHECKPOINT_FAILED
    assert result.failing_step_id == "s4_checkpoint"


async def test_unresolvable_target_is_hard_failure(tmp_path: Path):
    surface = FakeSurfaceAdapter()  # nothing registered at all
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.HARD_FAILURE
    assert result.code == OutcomeCode.TARGET_UNRESOLVABLE
    assert result.failing_step_id == "s2_fill"  # the first step whose target can't resolve


async def test_missing_required_input_is_hard_failure_invalid_input(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {}, surface)

    assert result.status == RunStatus.HARD_FAILURE
    assert result.code == OutcomeCode.INVALID_INPUT
    assert _business_action_calls(surface) == []  # no business action reached the surface
    # evidence capture is still allowed to run (and does, here) -- it is not
    # a "the surface was touched" signal, it's observability (Section 3.5).
    assert surface.calls == [("screenshot",)]
    assert len(result.evidence) == 1
    assert result.evidence[0].kind == "screenshot"


# -- POLICY_BLOCKED ----------------------------------------------------------

async def test_capability_target_domain_not_allowed_blocks_before_any_step(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    engine = _make_engine(tmp_path, allowed_domains=("some-other-host.example",))

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.POLICY_BLOCKED
    assert result.code == OutcomeCode.DOMAIN_NOT_ALLOWED
    assert _business_action_calls(surface) == []  # no business action reached the surface
    assert surface.calls == [("screenshot",)]  # evidence capture is still allowed to run
    assert len(result.evidence) == 1


async def test_prohibited_action_type_blocks_that_step(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    engine = _make_engine(
        tmp_path,
        allowed_actions=(ActionType.NAVIGATE,),  # FILL is not allowed
    )

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.POLICY_BLOCKED
    assert result.code == OutcomeCode.ACTION_TYPE_NOT_ALLOWED
    assert result.failing_step_id == "s2_fill"
    # navigate DID run -- policy check is per step. Recorded with its
    # fully-resolved absolute URL (Phase 6 bug fix: a surface with no
    # already-loaded page has no origin to resolve a bare relative route
    # against), not the artifact's raw "/search" route param.
    assert ("navigate", "http://127.0.0.1:8000/search") in surface.calls


async def test_irreversible_action_without_approval_is_policy_blocked(tmp_path: Path):
    from cua.artifact.schema import RiskLevel

    steps = [
        Step(
            step_id="close_account",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Close Account"),
            risk=RiskLevel.IRREVERSIBLE,
        ),
        Step(step_id="cp", action=ActionType.CHECKPOINT, checkpoint=CheckpointSpec(description="done", assertion=UrlMatchesCheck(pattern="/done"))),
    ]
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Close Account", [FakeElement()])
    engine = _make_engine(
        tmp_path, allowed_actions=(ActionType.CLICK, ActionType.CHECKPOINT)
    )

    result = await engine.run(_minimal_artifact(steps), {}, surface)

    assert result.status == RunStatus.POLICY_BLOCKED
    assert result.code == OutcomeCode.RISKY_ACTION_REQUIRES_APPROVAL
    assert _business_action_calls(surface) == []  # blocked before any business action reached the surface
    assert surface.calls == [("screenshot",)]  # evidence capture is still allowed to run
    assert len(result.evidence) == 1


async def test_redirect_escaping_allowlist_is_caught_after_the_click(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    _register_happy_path(surface)
    surface.script_redirect_on_next_click("http://evil.example.com/hijack")
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.status == RunStatus.POLICY_BLOCKED
    assert result.code == OutcomeCode.DOMAIN_NOT_ALLOWED
    assert result.failing_step_id == "s3_click"
    assert result.observed_state == "http://evil.example.com/hijack"


# -- bounded, deterministic retry --------------------------------------------

async def test_transient_not_found_recovers_within_bounded_retries(tmp_path: Path):
    steps = [
        Step(
            step_id="click_search",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Search"),
            retry=RetryPolicy(max_attempts=3, backoff_ms=1),
        ),
        Step(step_id="cp", action=ActionType.CHECKPOINT, checkpoint=CheckpointSpec(description="done", assertion=UrlMatchesCheck(pattern="/x"))),
    ]
    surface = FakeSurfaceAdapter()
    surface.register_role_name_flaky("button", "Search", [FakeElement()], fail_times=2)
    surface.script_checkpoint("/x", True)
    engine = _make_engine(tmp_path)

    result = await engine.run(_minimal_artifact(steps), {}, surface)

    assert result.status == RunStatus.SUCCESS
    assert len(result.recovery_attempts) == 2
    assert all(a.outcome == "failed" for a in result.recovery_attempts)


async def test_retry_is_bounded_and_eventually_hard_fails(tmp_path: Path):
    steps = [
        Step(
            step_id="click_search",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Search"),
            retry=RetryPolicy(max_attempts=2, backoff_ms=1),
        ),
        Step(step_id="cp", action=ActionType.CHECKPOINT, checkpoint=CheckpointSpec(description="done", assertion=UrlMatchesCheck(pattern="/x"))),
    ]
    surface = FakeSurfaceAdapter()
    surface.register_role_name_flaky("button", "Search", [FakeElement()], fail_times=5)  # never recovers within 2 attempts
    engine = _make_engine(tmp_path)

    result = await engine.run(_minimal_artifact(steps), {}, surface)

    assert result.status == RunStatus.HARD_FAILURE
    assert result.code == OutcomeCode.TARGET_UNRESOLVABLE
    assert len(result.recovery_attempts) == 1  # one failed retry recorded before the 2nd (final) attempt also failed


async def test_ambiguity_is_never_retried(tmp_path: Path):
    steps = [
        Step(
            step_id="click_search",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Search"),
            retry=RetryPolicy(max_attempts=5, backoff_ms=1),
        ),
        Step(step_id="cp", action=ActionType.CHECKPOINT, checkpoint=CheckpointSpec(description="done", assertion=UrlMatchesCheck(pattern="/x"))),
    ]
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement(), FakeElement()])
    engine = _make_engine(tmp_path)

    result = await engine.run(_minimal_artifact(steps), {}, surface)

    assert result.status == RunStatus.ESCALATION_REQUIRED
    assert result.recovery_attempts == []  # no retry attempts were made for an ambiguity
    assert _business_action_calls(surface) == []  # resolve() raised before click() was ever called
    assert surface.calls == [("screenshot",)]  # evidence capture is still allowed to run
    assert len(result.evidence) == 1


# -- explicit proof: policy/input/artifact validation never lets a business
# action through, but evidence capture and correct classification both
# still happen (this consolidates the pattern exercised piecemeal above
# into one direct assertion per the Phase 3 review requirements) ----------

async def test_blocked_runs_never_reach_the_surface_for_business_actions_but_are_still_evidenced_and_classified(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    # allowed_domains deliberately excludes the artifact's own target, so
    # this blocks at the very first upfront check, before any step exists.
    engine = _make_engine(tmp_path, allowed_domains=("not-this-host.example",))

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    # 1. no click/fill/navigate/select/read reached the surface
    assert _business_action_calls(surface) == []
    # 2. evidence capture (screenshot) still occurred
    assert surface.calls == [("screenshot",)]
    assert result.evidence and result.evidence[0].kind == "screenshot"
    # 3. the result is correctly classified, not a generic/opaque failure
    assert result.status == RunStatus.POLICY_BLOCKED
    assert result.code == OutcomeCode.DOMAIN_NOT_ALLOWED
    assert result.message  # a redacted, human-readable reason is present


# -- evidence -----------------------------------------------------------------

async def test_failure_captures_screenshot_evidence(tmp_path: Path):
    surface = FakeSurfaceAdapter()  # nothing registered -> TARGET_UNRESOLVABLE
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert len(result.evidence) == 1
    assert result.evidence[0].kind == "screenshot"
    assert Path(result.evidence[0].path).exists()


async def test_success_does_not_bother_capturing_evidence(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    _register_happy_path(surface)
    engine = _make_engine(tmp_path)

    result = await engine.run(_member_lookup_artifact(), {"member_id": "12345"}, surface)

    assert result.evidence == []
