"""Small, deterministic heuristics shared by the discovery loop (deciding
how to read a value live) and `ArtifactBuilder` (deciding how the
persisted `ExtractionSpec` should read the same value on replay) -- kept
in one place specifically so the two never drift apart from each other.
"""
from __future__ import annotations

import re

from cua.artifact.targets import TargetDescriptor

_INPUT_ROLES = frozenset({"textbox", "input", "combobox", "searchbox"})
_CURRENCY_RE = re.compile(r"^\$[\d,]+\.\d{2}$")


def infer_read_attribute(target: TargetDescriptor | None) -> str:
    """SurfaceAdapter.read()'s `attribute` argument: an input-shaped
    control's meaningful content is its current value; anything else
    (a status/label/cell) is its rendered text."""
    if target is not None and target.role and target.role.lower() in _INPUT_ROLES:
        return "value"
    return "text"


def infer_transform(raw_value: str) -> str:
    """Picks the narrowest closed transform (`cua.artifact.extraction.ExtractionSpec.transform`)
    that the observed value's shape actually supports -- never guesses a
    transform that would fail on the exact value discovery just saw."""
    if _CURRENCY_RE.match(raw_value.strip()):
        return "currency_to_float"
    if raw_value != raw_value.strip():
        return "strip"
    return "none"
