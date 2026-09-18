"""DiscoveryLoop tests -- all against FakeSurfaceAdapter + FakeModelAdapter,
no browser and no API key needed. These prove the OBSERVE -> DECIDE ->
VALIDATE -> POLICY CHECK -> ACT -> OBSERVE cycle actually enforces every
stopping condition and safety boundary Phase 4 requires, using a fully
scripted, deterministic model so nothing here depends on real model
non-determinism.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.loop import DiscoveryLoop
from cua.agent.trace import DiscoveryStopReason
from cua.artifact.targets import TargetDescriptor
from cua.models.base import ModelOutputError
from cua.observability.logger import StructuredLogger
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.surfaces.observation import InteractiveElement, Observation

from discovery_fixtures import BASE, build_member_lookup_fixture
from discovery_fixtures import obs as _obs
from fake_model import FakeModelAdapter
from fakes import FakeElement, FakeSurfaceAdapter


def _make_loop(surface: FakeSurfaceAdapter, model: FakeModelAdapter, tmp_path: Path, **overrides) -> DiscoveryLoop:
    policy = PolicyEngine(
        PolicyConfig(
            allowed_domains=("127.0.0.1",),
            allowed_actions=(
                ActionType.NAVIGATE, ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
                ActionType.READ_EXTRACT, ActionType.CHECKPOINT, ActionType.WAIT,
            ),
        )
    )
    logger = StructuredLogger(run_id="discovery-test", log_dir=tmp_path / "logs")
    evidence = DiscoveryEvidenceCapture(tmp_path / "evidence")
    defaults = dict(max_steps=10, overall_timeout_s=300, stuck_repeat_threshold=3, max_invalid_output_retries=2)
    defaults.update(overrides)
    return DiscoveryLoop(model, surface, policy, logger, evidence, **defaults)


# Calls that represent the surface actually being *acted on* -- as opposed
# to "observe" (perception, not action) and "screenshot" (evidence
# capture) -- same distinction Phase 3's test_replay_engine.py draws for
# ReplayEngine, applied here to the discovery loop.
_BUSINESS_ACTIONS = frozenset({"navigate", "click", "fill", "select", "read"})


def _business_calls(surface: FakeSurfaceAdapter) -> list[tuple]:
    return [call for call in surface.calls if call[0] in _BUSINESS_ACTIONS]


async def test_multi_step_discovery_reaches_goal(tmp_path: Path):
    surface, model = build_member_lookup_fixture()
    loop = _make_loop(surface, model, tmp_path)

    trace = await loop.run("Look up member 12345 and report their savings balance.")

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    executed = trace.successfully_executed_steps()
    assert [s.action.action for s in executed] == [
        ActionType.NAVIGATE, ActionType.FILL, ActionType.CLICK, ActionType.CHECKPOINT, ActionType.READ_EXTRACT, ActionType.FINISH,
    ]
    read_step = next(s for s in executed if s.action.action == ActionType.READ_EXTRACT)
    assert read_step.extracted_value == "$1,234.56"
    checkpoint_step = next(s for s in executed if s.action.action == ActionType.CHECKPOINT)
    assert checkpoint_step.checkpoint_assertion is not None
    assert checkpoint_step.checkpoint_assertion.type == "url_matches"
    assert checkpoint_step.checkpoint_assertion.pattern == "/results"
    assert len(model.calls) == 6  # one decide() per step, zero retries in the happy path


async def test_malformed_proposal_recovers_within_bounded_retries(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Go")]))
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Go")]))
    surface.register_role_name("button", "Go", [FakeElement()])
    model = FakeModelAdapter([
        ModelOutputError("simulated malformed tool call: missing required field 'action'"),
        AgentAction(action=ActionType.FINISH, reasoning="done"),
    ])
    loop = _make_loop(surface, model, tmp_path, max_invalid_output_retries=2)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert trace.steps[0].invalid_output_attempts == 1  # one rejected attempt before the valid one


async def test_malformed_proposal_exhausts_retries_and_escalates(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        ModelOutputError("bad 1"), ModelOutputError("bad 2"), ModelOutputError("bad 3"),
    ])
    loop = _make_loop(surface, model, tmp_path, max_invalid_output_retries=2)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.INVALID_OUTPUT_EXHAUSTED
    assert len(model.calls) == 3  # 1 initial + 2 retries, all exhausted
    assert "bad 3" in trace.stop_message


async def test_ungrounded_target_is_rejected_by_validate_not_executed(tmp_path: Path):
    """The model claims a control that was never in the observation --
    VALIDATE must reject it before POLICY CHECK or ACT ever see it."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Real Button")]))
    hallucinated = AgentAction(
        action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Nonexistent Button"),
        reasoning="click it",
    )
    model = FakeModelAdapter([hallucinated, hallucinated, hallucinated])
    loop = _make_loop(surface, model, tmp_path, max_invalid_output_retries=2)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.INVALID_OUTPUT_EXHAUSTED
    assert "not grounded" in trace.stop_message
    assert _business_calls(surface) == []  # never reached the surface at all


async def test_policy_blocked_proposal_stops_before_surface_is_touched(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Close Account")]))
    action = AgentAction(
        action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Close Account"),
        reasoning="close the account",
    )
    surface.register_role_name("button", "Close Account", [FakeElement()])
    model = FakeModelAdapter([action])
    loop = _make_loop(surface, model, tmp_path)  # IRREVERSIBLE risk requires approval by default

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.POLICY_BLOCKED
    assert trace.steps[-1].policy_decision is not None
    assert trace.steps[-1].policy_decision.requires_approval is True
    assert _business_calls(surface) == []  # no click reached the surface
    assert ("screenshot",) in surface.calls  # evidence capture still ran


# -- NAVIGATE resolution: the GOAL+TARGET fix for the genuine Phase 8
# POLICY_BLOCKED/host='' failure a real AnthropicModelAdapter run hit
# (attempt 1 against the local demo app -- see the Phase 8 commit history
# and REPORT.md for the full root-cause writeup). A fresh surface's
# current_url() is "about:blank", which is not a hierarchical URL a
# relative NAVIGATE route can be resolved against; these tests cover the
# resolution logic directly (fast, no browser) plus the handful of
# full-loop scenarios that need PolicyEngine actually involved. None of
# this weakens PolicyEngine or bypasses check_domain -- every case below
# still asserts the resolved URL is independently policy-checked. ---

def test_relative_navigate_resolves_against_trusted_target_base_url_when_current_url_has_no_anchor(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    model = FakeModelAdapter([])
    loop = _make_loop(surface, model, tmp_path, target_base_url="http://127.0.0.1:9000")

    resolved = loop._resolve_navigation_url("about:blank", "/search")

    assert resolved == "http://127.0.0.1:9000/search"


def test_absolute_url_is_returned_unchanged_regardless_of_target_base_url(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    model = FakeModelAdapter([])
    loop = _make_loop(surface, model, tmp_path, target_base_url="http://127.0.0.1:9000")

    resolved = loop._resolve_navigation_url("about:blank", "http://127.0.0.1:8000/search")

    assert resolved == "http://127.0.0.1:8000/search"


def test_relative_navigate_with_no_trusted_target_and_no_anchor_fails_safely_not_a_crash(tmp_path: Path):
    """The exact pre-fix reproduction: no target_base_url configured, no
    real page loaded yet -- must resolve to *something* PolicyEngine can
    safely reject (never raise, never silently pass through), matching
    the genuine 'host don't-know-the-target-yet' failure observed."""
    surface = FakeSurfaceAdapter()
    model = FakeModelAdapter([])
    loop = _make_loop(surface, model, tmp_path)  # no target_base_url

    resolved = loop._resolve_navigation_url("about:blank", "/search")

    assert not loop._is_absolute_http_url(resolved)
    decision = loop._policy.check_domain(resolved)
    assert decision.allowed is False
    assert "not in the allowed domain list" in decision.reason


async def test_full_loop_relative_navigate_with_target_base_url_reaches_goal(tmp_path: Path):
    """The full end-to-end regression proof: the exact scripted first
    action a real model proposed (a bare relative route, e.g. '/search',
    on a fresh session) now succeeds once target_base_url is configured
    -- reproducing the genuine Phase 8 failure and proving the fix."""
    surface = FakeSurfaceAdapter()
    member_id_el = InteractiveElement(role="textbox", accessible_name="Member ID")
    search_el = InteractiveElement(role="button", accessible_name="Search")
    balance_el = InteractiveElement(role="status", accessible_name="Savings Balance")
    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.register_role_name("status", "Savings Balance", [FakeElement(text="$1,234.56")])
    surface.script_observation(_obs("about:blank", []))  # the real, un-anchored first observation
    surface.script_observation(_obs(f"{BASE}/search", [member_id_el, search_el]))
    surface.script_observation(_obs(f"{BASE}/search", [member_id_el, search_el]))
    surface.script_observation(_obs(f"{BASE}/results", [balance_el]))
    surface.script_observation(_obs(f"{BASE}/results", [balance_el]))

    model = FakeModelAdapter([
        AgentAction(action=ActionType.NAVIGATE, params={"route": "/search"}, reasoning="Go to the search page."),
        AgentAction(action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
                    params={"value": "12345"}, reasoning="Enter the member id."),
        AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"),
                    reasoning="Submit the search."),
        AgentAction(action=ActionType.CHECKPOINT, params={"description": "results reached"}, reasoning="Confirm results."),
        AgentAction(action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
                    params={"output_name": "savings_balance"}, reasoning="Read the balance."),
        AgentAction(action=ActionType.FINISH, reasoning="Found the balance; goal achieved."),
    ])
    loop = _make_loop(surface, model, tmp_path, target_base_url=BASE)

    trace = await loop.run("Look up member 12345 and return their current savings balance.")

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert ("navigate", f"{BASE}/search") in surface.calls  # resolved against the trusted target, not left relative


async def test_absolute_disallowed_navigate_is_policy_blocked(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs("about:blank", []))
    action = AgentAction(action=ActionType.NAVIGATE, params={"url": "http://evil.example.com/"}, reasoning="go there")
    model = FakeModelAdapter([action])
    loop = _make_loop(surface, model, tmp_path, target_base_url=BASE)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.POLICY_BLOCKED
    assert "evil.example.com" in trace.stop_message
    assert _business_calls(surface) == []


def test_protocol_relative_escape_attempt_cannot_bypass_the_allowlist(tmp_path: Path):
    """A '//host/path' relative reference is a valid network-path
    reference per RFC 3986 -- urljoin correctly replaces the host, exactly
    as a browser would. Resolution is not the security boundary here;
    PolicyEngine.check_domain() is, and it still runs on whatever gets
    resolved, so this can never turn into an allowed navigation."""
    surface = FakeSurfaceAdapter()
    model = FakeModelAdapter([])
    loop = _make_loop(surface, model, tmp_path, target_base_url=BASE)

    resolved = loop._resolve_navigation_url("about:blank", "//evil.example.com/steal")

    assert resolved.startswith("http://evil.example.com")  # resolution behaves correctly per RFC 3986...
    decision = loop._policy.check_domain(resolved)
    assert decision.allowed is False  # ...but PolicyEngine still rejects it -- no bypass


async def test_redirect_to_disallowed_host_after_click_is_still_caught(tmp_path: Path):
    """Section 3.4's redirect warning, applied to discovery: even after a
    NAVIGATE/CLICK succeeds, the post-action check_domain() re-check
    (already in run(), unmodified by this fix) must still catch a
    surface-level redirect that escapes the allowlist."""
    surface = FakeSurfaceAdapter()
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.script_observation(_obs(f"{BASE}/search", [InteractiveElement(role="button", accessible_name="Search")]))
    surface.script_observation(_obs(f"{BASE}/search", [InteractiveElement(role="button", accessible_name="Search")]))
    surface.script_redirect_on_next_click("http://evil.example.com/hijack")
    action = AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"), reasoning="submit")
    model = FakeModelAdapter([action])
    loop = _make_loop(surface, model, tmp_path, target_base_url=BASE)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.POLICY_BLOCKED
    assert "escaped the allowlist" in trace.stop_message


async def test_max_steps_terminates_a_never_finishing_run(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    elements = [InteractiveElement(role="button", accessible_name=f"Btn{i}") for i in range(10)]
    for i in range(10):
        surface.script_observation(_obs(f"{BASE}/", [elements[i]]))
        surface.register_role_name("button", f"Btn{i}", [FakeElement()])
    actions = [
        AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name=f"Btn{i}"), reasoning="click")
        for i in range(10)
    ]
    model = FakeModelAdapter(actions)
    loop = _make_loop(surface, model, tmp_path, max_steps=5)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.MAX_STEPS
    assert len(trace.steps) == 5


async def test_overall_timeout_stops_discovery_deterministically(tmp_path: Path):
    """overall_timeout_s=0 must trip on the very first loop iteration --
    time.monotonic() - started is always > 0 by the time the check runs,
    so this is deterministic and instant, unlike a real 300s timeout test
    would be (Phase 7 item 4 explicitly asked for this to stay fast)."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Btn")]))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Btn"), reasoning="click"),
    ])
    loop = _make_loop(surface, model, tmp_path, overall_timeout_s=0)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.TIMEOUT
    assert len(trace.steps) == 0  # stopped before any step executed
    assert len(model.calls) == 0  # never even asked the model to decide


async def test_repeated_state_and_action_triggers_stuck_detection(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    obs = _obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Retry")])
    for _ in range(5):
        surface.script_observation(obs)
    surface.register_role_name("button", "Retry", [])  # never resolves -- always TargetNotFound
    action = AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Retry"), reasoning="try again")
    model = FakeModelAdapter([action, action, action, action, action])
    loop = _make_loop(surface, model, tmp_path, stuck_repeat_threshold=3)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.STUCK_STATE
    # 2 real (failed) attempts execute before the 3rd identical proposal is caught and stopped
    assert len(model.calls) == 3


async def test_request_human_stops_with_the_models_reason(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.REQUEST_HUMAN, params={"reason": "unexpected compliance-hold dialog"}, reasoning="need help")
    ])
    loop = _make_loop(surface, model, tmp_path)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.HUMAN_REQUESTED
    assert trace.stop_message == "unexpected compliance-hold dialog"


async def test_finish_stops_with_goal_achieved_and_no_surface_action(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([AgentAction(action=ActionType.FINISH, reasoning="already done")])
    loop = _make_loop(surface, model, tmp_path)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert _business_calls(surface) == []  # FINISH never touches the surface
    assert ("screenshot",) not in surface.calls  # no evidence capture needed on a clean success


async def test_unrecoverable_surface_error_stops_discovery(tmp_path: Path):
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Go")]))
    surface.register_role_name("button", "Go", [FakeElement()])
    surface.script_raise("click", RuntimeError("browser crashed"))
    action = AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Go"), reasoning="go")
    model = FakeModelAdapter([action])
    loop = _make_loop(surface, model, tmp_path)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.SURFACE_ERROR
    assert "browser crashed" in trace.stop_message


async def test_transient_target_not_found_is_reported_to_the_model_not_a_stop(tmp_path: Path):
    """A single failed action isn't automatically fatal -- unlike replay's
    bounded automatic retry, discovery feeds the failure back to the model
    (via history) and lets it decide, exactly once here before recovering."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Go")]))
    surface.script_observation(_obs(f"{BASE}/", [InteractiveElement(role="button", accessible_name="Go")]))
    surface.register_role_name_flaky("button", "Go", [FakeElement()], fail_times=1)
    action = AgentAction(action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Go"), reasoning="go")
    finish = AgentAction(action=ActionType.FINISH, reasoning="done")
    model = FakeModelAdapter([action, finish])
    loop = _make_loop(surface, model, tmp_path)

    trace = await loop.run("goal")

    assert trace.stop_reason == DiscoveryStopReason.GOAL_ACHIEVED
    assert trace.steps[0].execution_error is not None  # the first click's failure is on record
    assert trace.steps[0].executed is False


async def test_redaction_applies_to_persisted_discovery_logs(tmp_path: Path):
    """Secret-shaped text in a model's reasoning must never reach the log
    file in the clear (Section 5)."""
    surface = FakeSurfaceAdapter()
    surface.script_observation(_obs(f"{BASE}/", []))
    model = FakeModelAdapter([
        AgentAction(action=ActionType.FINISH, reasoning="using api_key=sk-should-not-appear-in-logs to finish")
    ])
    loop = _make_loop(surface, model, tmp_path)

    await loop.run("goal")

    log_path = tmp_path / "logs" / "discovery-test.jsonl"
    contents = log_path.read_text()
    assert "sk-should-not-appear-in-logs" not in contents
    assert "[REDACTED]" in contents
