"""AnthropicModelAdapter unit tests -- entirely against a mocked
`AsyncAnthropic` client, never a live API call. These are NOT genuine LLM
evidence (Phase 8 Section 1 is explicit about that); they exist to verify
the adapter's own contract -- client call shape, forced tool use, response
parsing, Pydantic validation of the model's output, error handling, and
the non-secret call-metadata this adapter now records -- independently of
whether a real credential is ever available in this environment.

Real SDK response types (`anthropic.types.Message`/`ToolUseBlock`/`Usage`)
are constructed directly rather than loose mocks/dicts, so a schema drift
in the installed `anthropic` package would show up here as a construction
error, not be silently masked by an overly-permissive fake.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from cua.actions import ActionType
from cua.models.anthropic_adapter import AnthropicModelAdapter, ModelCallMetadata
from cua.models.base import ModelOutputError
from cua.surfaces.observation import InteractiveElement, Observation

_OBS = Observation(
    url="http://127.0.0.1:8000/search",
    title="Member Search",
    interactive_elements=[
        InteractiveElement(role="textbox", accessible_name="Member ID", control_type="input"),
        InteractiveElement(role="button", accessible_name="Search", control_type="input"),
    ],
    visible_text_summary="Member Search",
    step_number=0,
)


def _tool_use_message(input_dict: dict, *, model="claude-x", stop_reason="tool_use", input_tokens=42, output_tokens=17) -> Message:
    block = ToolUseBlock(id="toolu_1", input=input_dict, name="propose_action", type="tool_use")
    return Message(
        id="msg_1", content=[block], model=model, role="assistant", type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens), stop_reason=stop_reason,
    )


def _mock_client(response: Message | Exception) -> AsyncMock:
    client = AsyncMock()
    if isinstance(response, Exception):
        client.messages.create = AsyncMock(side_effect=response)
    else:
        client.messages.create = AsyncMock(return_value=response)
    return client


async def test_decide_calls_the_sdk_with_forced_tool_use_and_a_bounded_timeout():
    client = _mock_client(_tool_use_message({
        "action": "click", "target": {"role": "button", "accessible_name": "Search"}, "reasoning": "submit the search",
    }))
    adapter = AnthropicModelAdapter(client, "claude-x", request_timeout_s=12.5)

    await adapter.decide(_OBS, ["goal: look up a member"])

    assert client.messages.create.await_count == 1
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-x"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "propose_action"}
    assert kwargs["tools"][0]["name"] == "propose_action"
    assert kwargs["timeout"] == 12.5  # Phase 8 fix: a hung request is now bounded, not left to the SDK default
    # the observation actually reached the prompt -- not a static/ignored payload
    assert "Member ID" in kwargs["messages"][0]["content"]
    assert "Search" in kwargs["messages"][0]["content"]


async def test_decide_parses_a_valid_tool_use_response_into_an_agent_action():
    client = _mock_client(_tool_use_message({
        "action": "fill", "target": {"role": "textbox", "accessible_name": "Member ID"},
        "params": {"value": "12345"}, "reasoning": "enter the member id",
    }))
    adapter = AnthropicModelAdapter(client, "claude-x")

    action = await adapter.decide(_OBS, [])

    assert action.action == ActionType.FILL
    assert action.target.accessible_name == "Member ID"
    assert action.params == {"value": "12345"}
    assert action.reasoning == "enter the member id"


async def test_decide_records_non_secret_call_metadata():
    client = _mock_client(_tool_use_message(
        {"action": "checkpoint", "params": {"description": "results reached"}, "reasoning": "confirm results page"},
        model="claude-real-model-id", stop_reason="tool_use", input_tokens=123, output_tokens=45,
    ))
    adapter = AnthropicModelAdapter(client, "claude-real-model-id")

    await adapter.decide(_OBS, [])

    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert isinstance(call, ModelCallMetadata)
    assert call.provider == "anthropic"
    assert call.model == "claude-real-model-id"
    assert call.stop_reason == "tool_use"
    assert call.input_tokens == 123
    assert call.output_tokens == 45
    assert call.action == "checkpoint"
    # timestamp is real and parseable, not a placeholder
    from datetime import datetime
    datetime.fromisoformat(call.timestamp)
    # no field on this dataclass could ever hold a prompt, a response body, or a header
    assert set(ModelCallMetadata.__dataclass_fields__) == {
        "timestamp", "provider", "model", "stop_reason", "input_tokens", "output_tokens", "action",
    }


async def test_decide_accumulates_one_metadata_entry_per_successful_call():
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=[
        _tool_use_message({"action": "navigate", "params": {"url": "http://127.0.0.1:8000/search"}, "reasoning": "go to search"}),
        _tool_use_message({"action": "finish", "reasoning": "done"}),
    ])
    adapter = AnthropicModelAdapter(client, "claude-x")

    await adapter.decide(_OBS, [])
    await adapter.decide(_OBS, [])

    assert len(adapter.calls) == 2
    assert [c.action for c in adapter.calls] == ["navigate", "finish"]


async def test_missing_tool_use_block_raises_model_output_error_not_a_crash():
    """The model called no tool at all (e.g. stopped on end_turn with plain
    text) -- must become ModelOutputError, never an unhandled exception or
    a guessed action."""
    text_only = Message(
        id="msg_2", content=[TextBlock(text="I'm not sure what to do.", type="text")],
        model="claude-x", role="assistant", type="message",
        usage=Usage(input_tokens=10, output_tokens=8), stop_reason="end_turn",
    )
    client = _mock_client(text_only)
    adapter = AnthropicModelAdapter(client, "claude-x")

    with pytest.raises(ModelOutputError, match="did not call"):
        await adapter.decide(_OBS, [])
    assert adapter.calls == []  # a rejected response must not be recorded as a successful decision


async def test_unrecognized_action_in_tool_input_raises_model_output_error():
    client = _mock_client(_tool_use_message({"action": "delete_everything", "reasoning": "oops"}))
    adapter = AnthropicModelAdapter(client, "claude-x")

    with pytest.raises(ModelOutputError, match="unrecognized action"):
        await adapter.decide(_OBS, [])


async def test_click_without_required_target_raises_model_output_error():
    """AgentAction's own Pydantic validation (CLICK requires a target) must
    surface as ModelOutputError, proving the adapter never bypasses
    AgentAction's schema just because the SDK's tool schema is satisfied."""
    client = _mock_client(_tool_use_message({"action": "click", "reasoning": "click something"}))
    adapter = AnthropicModelAdapter(client, "claude-x")

    with pytest.raises(ModelOutputError, match="schema validation"):
        await adapter.decide(_OBS, [])


async def test_invalid_target_shape_raises_model_output_error():
    client = _mock_client(_tool_use_message({
        "action": "click", "target": {"role": "button"},  # role without accessible_name/visible_text/nearby_anchor -- TargetDescriptor rejects this
        "reasoning": "click it",
    }))
    adapter = AnthropicModelAdapter(client, "claude-x")

    with pytest.raises(ModelOutputError, match="invalid target"):
        await adapter.decide(_OBS, [])


async def test_sdk_exception_is_wrapped_never_leaked_as_an_anthropic_type():
    """Any anthropic.* exception (rate limit, connection error, ...) must
    reach the discovery loop only as the provider-neutral ModelOutputError
    -- verified here with a generic exception standing in for any SDK
    error type, since the discovery loop must never need to know an
    Anthropic-specific exception class exists."""
    client = _mock_client(RuntimeError("connection reset by peer"))
    adapter = AnthropicModelAdapter(client, "claude-x")

    with pytest.raises(ModelOutputError, match="model call failed"):
        await adapter.decide(_OBS, [])
    assert adapter.calls == []


async def test_reasoning_is_truncated_never_a_persisted_chain_of_thought():
    long_reasoning = "x" * 5000
    client = _mock_client(_tool_use_message({"action": "finish", "reasoning": long_reasoning}))
    adapter = AnthropicModelAdapter(client, "claude-x")

    action = await adapter.decide(_OBS, [])

    assert len(action.reasoning) == 500  # bounded, matching _to_agent_action's [:500] truncation


async def test_prompt_never_includes_the_full_untruncated_history():
    """Only the last 10 history lines are ever sent -- proves the adapter
    does not accumulate and forward an ever-growing raw transcript."""
    client = _mock_client(_tool_use_message({"action": "finish", "reasoning": "done"}))
    adapter = AnthropicModelAdapter(client, "claude-x")
    history = [f"step {i}: did something" for i in range(1, 21)]  # 20 lines

    await adapter.decide(_OBS, history)

    sent_prompt = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "step 1:" not in sent_prompt
    assert "step 11:" in sent_prompt
    assert "step 20:" in sent_prompt


def test_default_request_timeout_is_bounded_not_unset():
    """Constructing with no explicit request_timeout_s must still produce
    a finite, sane default -- not None/unbounded (the bug this test guards
    against: relying silently on the SDK's own multi-minute default)."""
    adapter = AnthropicModelAdapter(AsyncMock(), "claude-x")
    assert adapter._request_timeout_s == 60.0
    assert 0 < adapter._request_timeout_s < 300
