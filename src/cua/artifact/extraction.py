"""ExtractionSpec: how a replay step turns UI state into a typed output.

Like `CheckpointAssertion`, `transform` is a closed set of named, reviewed
operations rather than an arbitrary expression -- the point of a
capability artifact is that a human reviewer can read exactly what it will
extract and how, without executing anything.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from cua.artifact.targets import TargetDescriptor


class ExtractionSpec(BaseModel):
    output_name: str
    target: TargetDescriptor
    attribute: Literal["text", "value", "href", "aria_label"] = "text"
    transform: Literal["none", "strip", "currency_to_float"] = "none"
