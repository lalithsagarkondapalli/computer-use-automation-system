"""Deterministic state/action fingerprinting for stuck-state detection.

Section 3.1/Phase 4 requires a "repeated-state/stuck detection" stopping
condition, and explicitly warns against comparing giant raw strings when a
compact normalized representation is more appropriate -- an
`Observation.visible_text_summary` can be up to 2000 characters and two
observations that are semantically identical are not always byte-identical
(whitespace, ordering). These functions normalize first, then hash.

Two independent fingerprints are combined, deliberately narrowly, by the
discovery loop (`cua.agent.loop`): a fingerprint of *what the surface
currently looks like* and a fingerprint of *what the model just proposed
against it*. Stuck detection triggers only when the exact same
(observation, proposal) pair repeats several times in a row -- not merely
"the URL didn't change," which would false-positive on any ordinary
multi-step interaction confined to one page (e.g. fill a field, then click
a button).
"""
from __future__ import annotations

import hashlib
import json

from cua.agent.actions import AgentAction
from cua.surfaces.observation import Observation


def fingerprint_observation(observation: Observation) -> str:
    """A short, order-independent hash of the parts of an Observation that
    represent "what is on screen," excluding step_number/screenshot_path
    (which change every step regardless of real progress) and excluding
    visible_text_summary's exact byte content (collapsed to length-bucket
    only, since two renders of the same state can differ in incidental
    whitespace).
    """
    elements_key = sorted(f"{e.role}:{e.accessible_name or ''}:{e.control_type or ''}" for e in observation.interactive_elements)
    text_bucket = len(observation.visible_text_summary.split())
    payload = json.dumps(
        {"url": observation.url, "title": observation.title, "elements": elements_key, "text_word_count": text_bucket},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint_action(action: AgentAction) -> str:
    """A short hash of a proposed action's identity -- action type, target
    signals, and params -- but never `reasoning` (free text that can vary
    harmlessly between otherwise-identical proposals and would defeat
    stuck detection if included).
    """
    target = action.target
    target_key = (
        None
        if target is None
        else {
            "role": target.role,
            "accessible_name": target.accessible_name,
            "visible_text": target.visible_text,
            "nearby_anchor": target.nearby_anchor,
            "dom_selector": target.dom_selector,
        }
    )
    payload = json.dumps(
        {"action": action.action.value, "target": target_key, "params": dict(sorted(action.params.items()))},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
