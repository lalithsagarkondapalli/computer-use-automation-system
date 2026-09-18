"""Checkpoint assertions: typed postconditions a replay step can verify.

Kept as a closed, discriminated union rather than an arbitrary expression
language on purpose -- an artifact must not be able to smuggle arbitrary
code into a check (Section 6's "no arbitrary code execution from model
responses" applies just as much to what gets persisted as to what gets
proposed live), and a closed set is something both a human reviewer and
the ReplayEngine can reason about exhaustively. "Checkpoint" in the
glossary sense: a condition you assert to confirm you actually reached the
expected state, rather than assuming a click worked.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from cua.artifact.targets import TargetDescriptor


class UrlMatchesCheck(BaseModel):
    type: Literal["url_matches"] = "url_matches"
    pattern: str  # substring match against the current URL


class ElementVisibleCheck(BaseModel):
    type: Literal["element_visible"] = "element_visible"
    target: TargetDescriptor


class ElementNotVisibleCheck(BaseModel):
    type: Literal["element_not_visible"] = "element_not_visible"
    target: TargetDescriptor


class TextPresentCheck(BaseModel):
    type: Literal["text_present"] = "text_present"
    text: str


CheckpointAssertion = Annotated[
    Union[UrlMatchesCheck, ElementVisibleCheck, ElementNotVisibleCheck, TextPresentCheck],
    Field(discriminator="type"),
]


class CheckpointSpec(BaseModel):
    description: str
    assertion: CheckpointAssertion
