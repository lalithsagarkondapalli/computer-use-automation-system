"""Synthetic, in-memory member data for the demo app.

No database. Every ID here is obviously synthetic (5 fake digits, a
"SYNTH-" name prefix, a placeholder balance) -- Phase 6's "clearly
synthetic status" requirement, not an accident of a small fixture.

Three canonical member IDs exercise three genuinely different UI states,
each produced by this module + the templates, never by anything inside
the automation system:
  12345 -- found, balance displayed immediately.
  00000 -- not found (and, deliberately, so is every other unlisted ID --
           a safe default rather than a silent 500).
  55555 -- found, but blocked behind a Compliance Hold interstitial until
           a human clears it on the live page.

A fourth ID, 67890, is a second plain "found" member -- added in Phase 7
solely so the full discovery -> artifact -> deterministic-replay contract
test (tests/test_full_contract_e2e.py) can replay with a genuinely
different valid input than the one used during discovery, rather than
re-running the exact literal value discovery happened to observe.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticMember:
    member_id: str
    name: str
    savings_balance: str


_MEMBERS: dict[str, SyntheticMember] = {
    "12345": SyntheticMember(member_id="12345", name="SYNTH-MEMBER-ALPHA (fictional)", savings_balance="$1,234.56"),
    "55555": SyntheticMember(member_id="55555", name="SYNTH-MEMBER-GAMMA (fictional)", savings_balance="$9,876.54"),
    "67890": SyntheticMember(member_id="67890", name="SYNTH-MEMBER-DELTA (fictional)", savings_balance="$542.10"),
}

# IDs that resolve to a real record but are gated by a compliance hold.
_COMPLIANCE_HOLD_IDS = frozenset({"55555"})


def lookup(member_id: str) -> tuple[str, SyntheticMember | None]:
    """Returns (status, member). status is one of "found", "not_found",
    "compliance_hold" -- the demo app's own classification, which the
    templates render as genuinely different DOM/UI states. Nothing about
    ReplayEngine or CheckpointEngine is aware this function exists."""
    member = _MEMBERS.get(member_id)
    if member is None:
        return "not_found", None
    if member_id in _COMPLIANCE_HOLD_IDS:
        return "compliance_hold", member
    return "found", member
