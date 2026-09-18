"""ModelAdapter: the only interface through which an LLM touches this
system.

Deliberately tiny -- one method, one bounded input, one typed output --
because that narrowness is what makes the model/replay boundary auditable.
`cua.replay` never imports this module (there is a test asserting exactly
that), and the concrete `AnthropicModelAdapter` (Phase 4) is the sole place
a provider SDK is imported, so swapping providers is a one-file change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cua.agent.actions import AgentAction
from cua.surfaces.observation import Observation


class ModelOutputError(Exception):
    """Raised by a ModelAdapter when the underlying model's response could
    not be turned into a valid AgentAction: malformed/missing tool call,
    a tool name the system never offered, or fields that fail AgentAction's
    own schema validation (Phase 4, Section 6 -- "invalid model output
    should be handled deliberately," never coerced into something
    executable). The discovery loop, not the adapter, decides what happens
    next (bounded retry, then escalate) -- an adapter must never fall back
    to guessing, retrying internally, or executing anything itself.
    """


class ModelAdapter(ABC):
    @abstractmethod
    async def decide(self, observation: Observation, history: list[str]) -> AgentAction:
        """Given the current observation and a short textual history of
        prior steps, return exactly one proposed AgentAction, or raise
        ModelOutputError if the underlying response cannot be turned into
        one.

        Must not execute anything itself and must not be given the ability
        to run arbitrary code -- the caller (the discovery loop) is solely
        responsible for validating (policy + schema) and executing the
        returned action.
        """
        raise NotImplementedError
