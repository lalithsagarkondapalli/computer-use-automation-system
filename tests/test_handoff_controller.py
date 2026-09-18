"""HandoffController unit tests: every valid ownership transition, the
important invalid ones, real blocking/unblocking via asyncio.Event (not
polling), resume validation, and both intervention bounds.
"""
from __future__ import annotations

import asyncio

import pytest

from cua.handoff.controller import ControlViolation, HandoffController, InterventionBudgetExhausted
from cua.handoff.models import InterventionRequest
from cua.handoff.state_machine import ControlState, InvalidTransition


def _request(**overrides) -> InterventionRequest:
    defaults = dict(
        run_id="run-1", mode="replay", reason_code="AMBIGUOUS_CONTROL",
        reason_message="two candidates matched", current_url="http://127.0.0.1:8000/search",
    )
    defaults.update(overrides)
    return InterventionRequest.create(**defaults)


# -- valid transitions -------------------------------------------------------

async def test_full_valid_cycle_with_successful_resume():
    controller = HandoffController()
    assert controller.state == ControlState.AUTOMATION_CONTROL

    await controller.escalate(_request())
    assert controller.state == ControlState.HUMAN_CONTROL
    assert [t.to_state for t in controller.transitions] == [ControlState.PAUSED_FOR_HUMAN, ControlState.HUMAN_CONTROL]

    controller.request_resume()
    assert controller.state == ControlState.RESUME_REQUESTED

    ok = await controller.await_resume(validate=None)
    assert ok is True
    assert controller.state == ControlState.AUTOMATION_CONTROL


async def test_resume_requested_can_bounce_back_to_paused_for_human():
    controller = HandoffController()
    await controller.escalate(_request())
    controller.request_resume()

    ok = await controller.await_resume(validate=lambda: _false())
    assert ok is False
    # bounced through PAUSED_FOR_HUMAN back to HUMAN_CONTROL, both logged
    assert [t.to_state for t in controller.transitions[-3:]] == [
        ControlState.RESUME_REQUESTED, ControlState.PAUSED_FOR_HUMAN, ControlState.HUMAN_CONTROL,
    ]
    assert controller.state == ControlState.HUMAN_CONTROL


async def _false() -> bool:
    return False


async def _true() -> bool:
    return True


# -- invalid transitions ------------------------------------------------------

def test_cannot_skip_pause_and_go_straight_to_human_control():
    from cua.handoff.state_machine import is_valid_transition

    assert not is_valid_transition(ControlState.AUTOMATION_CONTROL, ControlState.HUMAN_CONTROL)


def test_cannot_go_directly_from_human_control_to_automation_control():
    from cua.handoff.state_machine import is_valid_transition

    assert not is_valid_transition(ControlState.HUMAN_CONTROL, ControlState.AUTOMATION_CONTROL)


def test_cannot_request_resume_from_paused_for_human():
    from cua.handoff.state_machine import is_valid_transition

    assert not is_valid_transition(ControlState.PAUSED_FOR_HUMAN, ControlState.RESUME_REQUESTED)


async def test_calling_request_resume_twice_in_a_row_is_rejected():
    controller = HandoffController()
    await controller.escalate(_request())
    controller.request_resume()
    with pytest.raises(InvalidTransition):
        controller.request_resume()  # already RESUME_REQUESTED; HUMAN_CONTROL -> RESUME_REQUESTED again is not HUMAN_CONTROL's current state


async def test_escalating_while_already_in_human_control_is_rejected():
    controller = HandoffController()
    await controller.escalate(_request())
    with pytest.raises(InvalidTransition):
        await controller.escalate(_request())  # AUTOMATION_CONTROL -> PAUSED_FOR_HUMAN is invalid from HUMAN_CONTROL


# -- "impossible by construction" while HUMAN_CONTROL is active -------------

async def test_automation_action_is_rejected_during_human_control():
    controller = HandoffController()
    await controller.escalate(_request())
    with pytest.raises(ControlViolation):
        controller.assert_automation_control()


def test_automation_action_is_permitted_in_automation_control():
    controller = HandoffController()
    controller.assert_automation_control()  # does not raise


# -- real blocking, not polling ----------------------------------------------

async def test_pause_actually_blocks_a_waiting_automation_coroutine():
    controller = HandoffController()
    await controller.escalate(_request())

    waiter = asyncio.ensure_future(controller.wait_until_automation_control())
    await asyncio.sleep(0)  # let it start waiting
    assert not waiter.done()  # still blocked -- no action has been let through

    controller.request_resume()
    await controller.await_resume(validate=None)
    await asyncio.wait_for(waiter, timeout=1)
    assert waiter.done()


async def test_resume_unblocks_a_waiting_automation_coroutine_immediately():
    controller = HandoffController()
    await controller.escalate(_request())
    waiter = asyncio.ensure_future(controller.wait_until_automation_control())
    controller.request_resume()
    ok = await controller.await_resume(validate=None)
    assert ok is True
    await asyncio.wait_for(waiter, timeout=1)  # would hang forever if the gate weren't real


# -- bounded resume attempts (escalate_and_wait) -----------------------------

async def test_escalate_and_wait_gives_up_after_bounded_failed_resume_attempts():
    controller = HandoffController(max_resume_attempts=2)
    attempts = 0

    async def always_fails() -> bool:
        nonlocal attempts
        attempts += 1
        return False

    async def click_resume_soon() -> None:
        for _ in range(2):
            await asyncio.sleep(0)
            controller.request_resume()
            await asyncio.sleep(0)

    result, _ = await asyncio.gather(
        controller.escalate_and_wait(_request(), validate=always_fails),
        click_resume_soon(),
    )
    assert result is False
    assert attempts == 2  # bounded, not infinite
    assert controller.current_intervention.resume_attempts == 2


async def test_escalate_and_wait_succeeds_once_validation_passes():
    controller = HandoffController(max_resume_attempts=3)

    async def click_resume_once() -> None:
        await asyncio.sleep(0)
        controller.request_resume()

    result, _ = await asyncio.gather(
        controller.escalate_and_wait(_request(), validate=_true),
        click_resume_once(),
    )
    assert result is True
    assert controller.state == ControlState.AUTOMATION_CONTROL


# -- bounded distinct interventions ------------------------------------------

async def test_total_intervention_budget_is_enforced_across_distinct_escalations():
    controller = HandoffController(max_interventions=1)
    await controller.escalate(_request())
    controller.request_resume()
    await controller.await_resume(validate=None)  # first intervention resolved, budget now spent

    with pytest.raises(InterventionBudgetExhausted):
        await controller.escalate(_request())


# -- InterventionRequest construction + redaction ----------------------------

def test_intervention_request_redacts_secret_shaped_fields():
    request = _request(reason_message="blocked because api_key=sk-should-not-leak was rejected")
    assert "sk-should-not-leak" not in request.reason_message
    assert "[REDACTED]" in request.reason_message


def test_intervention_request_carries_expected_context():
    request = InterventionRequest.create(
        run_id="run-42", mode="discovery", reason_code="STUCK_STATE",
        reason_message="repeated the same click 3 times", current_url="http://127.0.0.1:8000/search",
        capability_id="member_lookup_balance", capability_version="1.0.0",
        goal="find the balance", step_id="4", expected_state="results page", observed_state="still on search page",
    )
    assert request.intervention_id
    assert request.run_id == "run-42"
    assert request.mode == "discovery"
    assert request.capability_id == "member_lookup_balance"
    assert request.control_owner == ControlState.AUTOMATION_CONTROL
