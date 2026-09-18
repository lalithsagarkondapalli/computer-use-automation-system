"""Shared discovery-test fixture: a scripted member_lookup_balance
discovery run against FakeSurfaceAdapter/FakeModelAdapter.

Mirrors the same capability tests/factories.py hand-authors for replay
tests, specifically so a reader can compare "the artifact discovery
actually produced" against "the artifact a human would have hand-written"
for the identical capability (see test_discovery_to_replay_integration.py).
Centralized here so test_discovery_loop.py, test_artifact_builder.py, and
the integration test all exercise the exact same scripted run rather than
three slightly-different hand-rolled variants drifting apart.
"""
from __future__ import annotations

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.artifact.targets import TargetDescriptor
from cua.surfaces.observation import InteractiveElement, Observation

from fake_model import FakeModelAdapter
from fakes import FakeElement, FakeSurfaceAdapter

BASE = "http://127.0.0.1:8000"


def obs(url: str, elements: list[InteractiveElement], text: str = "") -> Observation:
    return Observation(url=url, title="", interactive_elements=elements, visible_text_summary=text, step_number=0)


def build_member_lookup_fixture(*, member_id: str = "12345", balance: str = "$1,234.56") -> tuple[FakeSurfaceAdapter, FakeModelAdapter]:
    """A full, scripted member_lookup_balance happy path: navigate to the
    search page, fill in the member id, click Search, confirm the results
    checkpoint, read the balance, finish."""
    surface = FakeSurfaceAdapter()
    member_id_el = InteractiveElement(role="textbox", accessible_name="Member ID")
    search_el = InteractiveElement(role="button", accessible_name="Search")
    balance_el = InteractiveElement(role="status", accessible_name="Savings Balance")

    surface.register_role_name("textbox", "Member ID", [FakeElement()])
    surface.register_role_name("button", "Search", [FakeElement()])
    surface.register_role_name("status", "Savings Balance", [FakeElement(text=balance)])

    obs0 = obs(f"{BASE}/", [], "Welcome to CU Ops Console")
    obs1 = obs(f"{BASE}/search", [member_id_el, search_el], "Member ID Search")
    obs2 = obs1
    obs3 = obs(f"{BASE}/results", [balance_el], f"Savings Balance {balance}")
    obs4 = obs3
    obs5 = obs3
    for o in (obs0, obs1, obs2, obs3, obs4, obs5):
        surface.script_observation(o)

    script = [
        AgentAction(action=ActionType.NAVIGATE, params={"route": "/search"}, reasoning="Go to the search page."),
        AgentAction(
            action=ActionType.FILL, target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
            params={"value": member_id}, reasoning="Enter the member id.",
        ),
        AgentAction(
            action=ActionType.CLICK, target=TargetDescriptor(role="button", accessible_name="Search"),
            reasoning="Submit the search.",
        ),
        AgentAction(action=ActionType.CHECKPOINT, params={"description": "results page reached"}, reasoning="Confirm we reached results."),
        AgentAction(
            action=ActionType.READ_EXTRACT, target=TargetDescriptor(role="status", accessible_name="Savings Balance"),
            params={"output_name": "savings_balance"}, reasoning="Read the balance.",
        ),
        AgentAction(action=ActionType.FINISH, reasoning="Found the balance; goal achieved."),
    ]
    model = FakeModelAdapter(script)
    return surface, model
