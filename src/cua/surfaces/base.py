"""SurfaceAdapter: the boundary between "how we perceive and act on a
screen" and everything else.

Playwright is one implementation of this (`PlaywrightWebSurface`, added in
Phase 3) -- nothing in `cua.agent`, `cua.replay`, or `cua.artifact` imports
Playwright directly. That is what lets the same recorded-capability concept
extend, in design if not in this build, to a legacy DOM-poor web app, an
accessibility-tree-only surface, or a native desktop application (see
REPORT.md Section 4) without touching those layers -- only a new
SurfaceAdapter implementation would be needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cua.artifact.checks import CheckpointAssertion
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.observation import Observation


@dataclass(frozen=True)
class ResolvedTarget:
    """One concrete match for a TargetDescriptor, with the strategy that found it.

    `handle` is surface-specific (a Playwright Locator today) and opaque
    outside the adapter that produced it -- callers never inspect it, only
    pass it back into the same adapter's click/fill/select/read methods.
    """

    handle: object
    strategy: str


class TargetNotFound(Exception):
    """No candidate matched the descriptor under any strategy in strategy_order."""


class TargetAmbiguous(Exception):
    """More than one candidate matched at the strategy that produced a match.

    Never resolved by picking one arbitrarily (Section 8) -- callers must
    treat this as ESCALATION_REQUIRED / AMBIGUOUS_CONTROL, not silently
    guess.
    """

    def __init__(self, count: int, strategy: str):
        super().__init__(f"{count} candidates matched via {strategy!r}")
        self.count = count
        self.strategy = strategy


class SurfaceAdapter(ABC):
    """Everything the replay engine and discovery loop need from a live surface."""

    @abstractmethod
    async def navigate(self, url_or_route: str) -> None: ...

    @abstractmethod
    async def resolve(self, target: TargetDescriptor) -> ResolvedTarget:
        """Resolve a descriptor to exactly one control.

        Implementations must try `target.strategy_order` in order and stop
        at the first strategy that yields exactly one match. Raises
        `TargetNotFound` if no strategy matches anything, or
        `TargetAmbiguous` if the first strategy to match anything matches
        more than one candidate.
        """
        ...

    @abstractmethod
    async def click(self, resolved: ResolvedTarget) -> None: ...

    @abstractmethod
    async def fill(self, resolved: ResolvedTarget, value: str) -> None: ...

    @abstractmethod
    async def select(self, resolved: ResolvedTarget, value: str) -> None: ...

    @abstractmethod
    async def read(self, resolved: ResolvedTarget, attribute: str) -> str: ...

    @abstractmethod
    async def wait_for(self, target: TargetDescriptor, timeout_ms: int) -> bool:
        """Wait for a target to become resolvable.

        Returns False on timeout rather than raising -- callers (the
        ReplayEngine) decide what a timeout means in context (retry,
        RECOVERABLE, or HARD_FAILURE), so this method itself stays a
        boolean fact, not a policy decision.
        """
        ...

    @abstractmethod
    async def wait_for_checkpoint(self, assertion: CheckpointAssertion, timeout_ms: int) -> bool:
        """Wait for a checkpoint assertion (Section 3.3's "postcondition")
        to hold, e.g. a URL pattern, an element becoming visible/hidden, or
        text appearing. Returns False on timeout, never raises for a mere
        timeout -- the ReplayEngine's CheckpointEngine decides what a
        failed checkpoint means (HARD_FAILURE, or a declared business
        outcome/escalation via a ConditionProbe). This is a distinct method
        from `resolve` because a checkpoint can assert something that is
        not "a control exists" (e.g. a URL), and because it must use each
        surface's own native, event-driven waiting where one exists,
        rather than the resolver's single-shot count check.
        """
        ...

    @abstractmethod
    async def current_url(self) -> str: ...

    @abstractmethod
    async def observe(self, step_number: int) -> Observation:
        """Produce the bounded Observation given to the discovery model."""
        ...

    @abstractmethod
    async def screenshot(self) -> bytes: ...

    @abstractmethod
    async def close(self) -> None: ...
