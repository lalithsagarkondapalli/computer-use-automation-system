"""Turn a model's plain-language checkpoint claim into a verifiable
`CheckpointAssertion`, deterministically -- never by asking the model to
emit the assertion itself.

Why: a checkpoint's whole point (Section 3.2/3.3) is that a human -- and
`ReplayEngine`'s `CheckpointEngine` -- can mechanically verify it later
without re-running any judgment. If the model were allowed to author the
assertion directly (e.g. hand us a URL pattern or a CSS-shaped string), a
hallucinated or overly specific pattern could sail through undetected,
and Section 6's "no arbitrary code/selectors from model output" would be
violated in spirit even if not in literal syntax. Instead, the model may
only *claim* a checkpoint with a short description (via a CHECKPOINT
`AgentAction`, params={"description": ...}); this module looks at what
*actually, structurally* changed between two real `Observation`s the
system itself captured, and picks the first assertion type below that the
observed change actually supports. If nothing changed in a way any of
these can express, the claim is rejected (returns None) -- the discovery
loop then treats it as an invalid proposal, not a silent no-op checkpoint.
"""
from __future__ import annotations

from urllib.parse import urlparse

from cua.artifact.checks import CheckpointAssertion, ElementVisibleCheck, TextPresentCheck, UrlMatchesCheck
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.observation import Observation

_MIN_DESCRIPTION_MATCH_LEN = 4  # avoid matching on trivially short/common words


def synthesize_checkpoint(description: str, before: Observation, after: Observation) -> CheckpointAssertion | None:
    """`before` is the observation captured immediately before the most
    recent state-changing action; `after` is the observation the model is
    looking at right now (checkpoint itself never changes state, so
    "after" here means "current," not "after the checkpoint")."""
    before_path = urlparse(before.url).path
    after_path = urlparse(after.url).path
    if after_path and after_path != before_path:
        return UrlMatchesCheck(pattern=after_path)

    before_elements = {(e.role, e.accessible_name) for e in before.interactive_elements}
    for element in after.interactive_elements:
        key = (element.role, element.accessible_name)
        if key not in before_elements and element.accessible_name:
            return ElementVisibleCheck(target=TargetDescriptor(role=element.role, accessible_name=element.accessible_name))

    normalized = description.strip()
    if len(normalized) >= _MIN_DESCRIPTION_MATCH_LEN and normalized.lower() in after.visible_text_summary.lower():
        return TextPresentCheck(text=normalized)

    return None
