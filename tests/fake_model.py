"""FakeModelAdapter: a deterministic, scripted stand-in for a real LLM,
used by every Phase 4 test that doesn't require an API key.

Takes a fixed list of "moves" up front and returns them one at a time,
ignoring the actual Observation/history content unless a move is a
callable (for the few tests that need a decision to depend on what the
surface currently shows, e.g. reacting to a failed action). This is a
test double for `ModelAdapter`, not a toy implementation of one -- real
non-determinism (an actual model's variability) is exactly what these
tests must NOT depend on.
"""
from __future__ import annotations

from collections.abc import Callable

from cua.agent.actions import AgentAction
from cua.models.base import ModelAdapter, ModelOutputError
from cua.surfaces.observation import Observation

Move = AgentAction | Exception | Callable[[Observation, list], AgentAction]


class FakeModelAdapter(ModelAdapter):
    def __init__(self, script: list[Move]):
        self._script = list(script)
        self.calls: list[tuple[Observation, list[str]]] = []

    async def decide(self, observation: Observation, history: list[str]) -> AgentAction:
        self.calls.append((observation, list(history)))
        if not self._script:
            raise ModelOutputError("FakeModelAdapter script exhausted -- test scripted too few moves")
        move = self._script.pop(0)
        if isinstance(move, Exception):
            raise move
        if callable(move):
            return move(observation, history)
        return move
