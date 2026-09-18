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


class ModelAdapter(ABC):
    @abstractmethod
    async def decide(self, observation: Observation, history: list[str]) -> AgentAction:
        """Given the current observation and a short textual history of
        prior steps, return exactly one proposed AgentAction.

        Must not execute anything itself and must not be given the ability
        to run arbitrary code -- the caller (the discovery loop) is solely
        responsible for validating (policy + schema) and executing the
        returned action.
        """
        raise NotImplementedError
