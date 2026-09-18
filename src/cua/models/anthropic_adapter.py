"""AnthropicModelAdapter: the concrete, provider-specific `ModelAdapter`.

This is the ONLY module in the repository that imports the `anthropic`
package (see tests/test_discovery_replay_isolation.py). Everything else --
the discovery loop, the artifact builder, all of replay -- talks to a
model only through `ModelAdapter.decide()`, so swapping providers (or
adding a second one) is a one-file change with no ripple effect.

The model is given exactly one tool, `propose_action`, whose JSON schema
mirrors `AgentAction`'s closed vocabulary (`cua.actions.ActionType`) --
`tool_choice` forces the model to call it, so its response is never free
prose to parse. No other tool is offered: no code execution, no shell,
no raw HTTP, no direct Playwright access (Section 6's "the model proposes,
the system executes" is enforced structurally here, not just by
convention -- there is nothing else this model could call). A malformed
or schema-violating response -- wrong tool, missing tool call, fields that
fail `AgentAction`'s own validation -- is turned into `ModelOutputError`
and handed back to the discovery loop, never guessed at or coerced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import ToolUseBlock

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.artifact.targets import TargetDescriptor
from cua.models.base import ModelAdapter, ModelOutputError
from cua.surfaces.observation import Observation


@dataclass(frozen=True)
class ModelCallMetadata:
    """Non-secret metadata about one real `decide()` call -- Phase 8's
    evidence requirement (Section 10: "record provider, model identifier,
    timestamp, number of decision calls, stop reason, token usage if
    exposed"). Deliberately holds no request/response body, no
    Authorization data, and no hidden model reasoning -- only the small
    set of fields a reviewer needs to establish a call really reached the
    provider. `AnthropicModelAdapter.calls` (below) is a list of these,
    mirroring `FakeModelAdapter.calls`'s existing test-introspection
    convention, so evidence-generation code and tests can both inspect
    call count/metadata without changing `ModelAdapter`'s interface."""

    timestamp: str
    provider: str
    model: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    action: str


_TOOL_NAME = "propose_action"

_ACTION_TOOL: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Propose exactly one next action against the current screen. You may only "
        "propose an action from this closed vocabulary -- you can never run "
        "arbitrary code, shell commands, or direct browser/HTTP calls."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [a.value for a in ActionType],
                "description": "The single action to take next.",
            },
            "target": {
                "type": ["object", "null"],
                "description": (
                    "Which control this action applies to, described only by signals "
                    "actually visible in the observation below -- never invent a CSS "
                    "selector or a control you were not shown."
                ),
                "properties": {
                    "role": {"type": ["string", "null"]},
                    "accessible_name": {"type": ["string", "null"]},
                    "visible_text": {"type": ["string", "null"]},
                    "nearby_anchor": {
                        "type": ["string", "null"],
                        "description": "A label/anchor text near the control, if that is how you identified it.",
                    },
                },
            },
            "params": {
                "type": "object",
                "description": (
                    "Action-specific parameters as strings: {'value': ...} for fill/select, "
                    "{'route': ...} or {'url': ...} for navigate, {'reason': ...} for "
                    "request_human, {'output_name': ...} for read_extract, "
                    "{'description': ...} for checkpoint."
                ),
                "additionalProperties": {"type": "string"},
            },
            "reasoning": {
                "type": "string",
                "description": "One short, action-oriented sentence: why this action, right now. Not a chain of thought.",
            },
        },
        "required": ["action", "reasoning"],
    },
}


class AnthropicModelAdapter(ModelAdapter):
    def __init__(
        self,
        client: AsyncAnthropic,
        model_name: str,
        *,
        max_tokens: int = 1024,
        request_timeout_s: float = 60.0,
    ):
        """`request_timeout_s` bounds a single `messages.create()` call
        (Phase 8 adapter audit: previously unset, so a single hung request
        fell back to the SDK's own multi-minute default and could not be
        interrupted by DiscoveryLoop's overall_timeout_s, which is only
        checked between steps, never during an in-flight call). 60s is
        generous for a single small structured-tool-use call and still
        bounded well under a typical discovery run's overall timeout."""
        self._client = client
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._request_timeout_s = request_timeout_s
        # Non-secret call metadata for Phase 8 evidence (Section 10) --
        # never the prompt, the response body, or any request header.
        self.calls: list[ModelCallMetadata] = []

    async def decide(self, observation: Observation, history: list[str]) -> AgentAction:
        prompt = self._build_prompt(observation, history)
        try:
            response = await self._client.messages.create(
                model=self._model_name,
                max_tokens=self._max_tokens,
                tools=[_ACTION_TOOL],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[{"role": "user", "content": prompt}],
                timeout=self._request_timeout_s,
            )
        except Exception as exc:
            # Any anthropic.* exception (APIError, RateLimitError, ...) is
            # deliberately caught broadly and re-raised as the provider-
            # neutral ModelOutputError -- nothing above this adapter should
            # ever need to know an Anthropic-specific exception type exists.
            raise ModelOutputError(f"model call failed: {exc}") from exc

        tool_use = next(
            (block for block in response.content if isinstance(block, ToolUseBlock) and block.name == _TOOL_NAME),
            None,
        )
        if tool_use is None:
            raise ModelOutputError(f"model did not call {_TOOL_NAME!r} (stop_reason={response.stop_reason!r})")

        action = self._to_agent_action(tool_use.input)
        usage = getattr(response, "usage", None)
        self.calls.append(ModelCallMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider="anthropic",
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            action=action.action.value,
        ))
        return action

    def _to_agent_action(self, raw: dict[str, Any]) -> AgentAction:
        try:
            action = ActionType(raw["action"])
        except (KeyError, ValueError) as exc:
            raise ModelOutputError(f"model proposed an unrecognized action: {raw.get('action')!r}") from exc

        target: TargetDescriptor | None = None
        raw_target = raw.get("target")
        if raw_target:
            try:
                target = TargetDescriptor(
                    role=raw_target.get("role"),
                    accessible_name=raw_target.get("accessible_name"),
                    visible_text=raw_target.get("visible_text"),
                    nearby_anchor=raw_target.get("nearby_anchor"),
                )
            except Exception as exc:
                raise ModelOutputError(f"model proposed an invalid target: {exc}") from exc

        try:
            return AgentAction(
                action=action,
                target=target,
                params={str(k): str(v) for k, v in (raw.get("params") or {}).items()},
                reasoning=str(raw.get("reasoning", ""))[:500],
            )
        except Exception as exc:
            raise ModelOutputError(f"model proposal failed schema validation: {exc}") from exc

    def _build_prompt(self, observation: Observation, history: list[str]) -> str:
        elements = "\n".join(
            f"- role={element.role!r} accessible_name={element.accessible_name!r} control_type={element.control_type!r}"
            for element in observation.interactive_elements
        ) or "(none)"
        # Bounded history window -- never the full transcript, and never
        # hidden chain-of-thought (see cua.agent.actions.AgentAction docstring).
        history_block = "\n".join(history[-10:])
        return (
            f"{history_block}\n\n"
            "Current screen:\n"
            f"URL: {observation.url}\n"
            f"Title: {observation.title}\n"
            f"Interactive elements:\n{elements}\n"
            f"Visible text (truncated): {observation.visible_text_summary[:1000]}\n\n"
            f"Call {_TOOL_NAME} with exactly one next action."
        )
