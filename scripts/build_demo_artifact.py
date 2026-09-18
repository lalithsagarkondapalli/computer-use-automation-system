"""Builds and saves the member_lookup_balance capability artifact for the
Phase 6 demo app. Hand-authored (mirroring what ArtifactBuilder would
produce from a real discovery run against this same app), then validated
by construction through the real CapabilityArtifact schema and saved
through the real cua.artifact.store -- not a fabricated/hand-written YAML
file. Run with:

    python scripts/build_demo_artifact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cua.actions import ActionType
from cua.artifact.checks import CheckpointSpec, ConditionProbe, ElementVisibleCheck, TextPresentCheck
from cua.artifact.extraction import ExtractionSpec
from cua.artifact.schema import (
    CapabilityArtifact,
    InputParam,
    OutputParam,
    ParamType,
    RiskLevel,
    Step,
    TargetIdentity,
)
from cua.artifact.store import save_artifact
from cua.artifact.targets import TargetDescriptor

BALANCE_TARGET = TargetDescriptor(role="status", accessible_name="Savings Balance")

STEPS = [
    Step(step_id="s1_navigate", action=ActionType.NAVIGATE, params={"route": "/search"}, risk=RiskLevel.SAFE),
    Step(
        step_id="s2_fill",
        action=ActionType.FILL,
        target=TargetDescriptor(role="textbox", accessible_name="Member ID"),
        params={"value": "{{member_id}}"},
        risk=RiskLevel.SAFE,
    ),
    Step(
        step_id="s3_click",
        action=ActionType.CLICK,
        target=TargetDescriptor(role="button", accessible_name="Search"),
        risk=RiskLevel.SAFE,
    ),
    Step(
        step_id="s4_checkpoint",
        action=ActionType.CHECKPOINT,
        checkpoint=CheckpointSpec(
            description="member record retrieved and balance visible",
            assertion=ElementVisibleCheck(target=BALANCE_TARGET),
            probes=[
                ConditionProbe(
                    description="member not found",
                    assertion=TextPresentCheck(text="No member found"),
                    status="BUSINESS_OUTCOME",
                    code="CUSTOMER_NOT_FOUND",
                ),
                ConditionProbe(
                    description="compliance hold interstitial blocking the record",
                    assertion=ElementVisibleCheck(target=TargetDescriptor(role="alertdialog", accessible_name="Compliance Hold")),
                    status="ESCALATION_REQUIRED",
                    code="UNKNOWN_DIALOG",
                ),
            ],
        ),
        timeout_ms=5_000,
        risk=RiskLevel.SAFE,
    ),
    Step(
        step_id="s5_extract",
        action=ActionType.READ_EXTRACT,
        target=BALANCE_TARGET,
        extraction=ExtractionSpec(output_name="savings_balance", target=BALANCE_TARGET, attribute="text", transform="currency_to_float"),
        risk=RiskLevel.SAFE,
    ),
]

ARTIFACT = CapabilityArtifact(
    capability_id="member_lookup_balance",
    capability_version="1.0.0",
    name="Look up member and read savings balance",
    description=(
        "Search the CU Ops Console demo app for a member by ID and extract their "
        "current savings balance. Business outcome (member not found) and an "
        "unexpected Compliance Hold interstitial (requiring human handoff) are "
        "both classified via checkpoint probes, not hardcoded in ReplayEngine."
    ),
    target=TargetIdentity(
        app_name="CU Ops Console",
        vendor="internal-demo",
        product="cu-ops-console",
        base_url_pattern="http://127.0.0.1:8000",
    ),
    risk=RiskLevel.SAFE,
    inputs=[
        InputParam(name="member_id", type=ParamType.STRING, description="5-digit synthetic member identifier", example="12345"),
    ],
    outputs=[
        OutputParam(name="savings_balance", type=ParamType.FLOAT, description="Current savings balance, parsed from currency text"),
    ],
    steps=STEPS,
    created_by="discovery_agent",
)


if __name__ == "__main__":
    path = save_artifact(ARTIFACT, REPO_ROOT / "artifacts")
    print(f"saved {path}")
