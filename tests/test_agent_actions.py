import pytest
from pydantic import ValidationError

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.artifact.targets import TargetDescriptor


def test_click_requires_target():
    with pytest.raises(ValidationError):
        AgentAction(action=ActionType.CLICK, reasoning="need to click search")


def test_fill_requires_value_param():
    with pytest.raises(ValidationError):
        AgentAction(
            action=ActionType.FILL,
            target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
            reasoning="typing the member id",
        )


def test_valid_fill():
    AgentAction(
        action=ActionType.FILL,
        target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
        params={"value": "{{member_id}}"},
        reasoning="typing the member id into the search box",
    )


def test_finish_needs_no_target():
    AgentAction(action=ActionType.FINISH, reasoning="goal achieved: balance extracted")


def test_request_human_requires_reason_param():
    with pytest.raises(ValidationError):
        AgentAction(action=ActionType.REQUEST_HUMAN, reasoning="stuck", params={})


def test_request_human_with_reason_is_valid():
    AgentAction(
        action=ActionType.REQUEST_HUMAN,
        reasoning="an unexpected modal appeared that is not part of any known flow",
        params={"reason": "unknown_dialog"},
    )


def test_valid_navigate():
    AgentAction(action=ActionType.NAVIGATE, params={"route": "/search"}, reasoning="go to the search page")
