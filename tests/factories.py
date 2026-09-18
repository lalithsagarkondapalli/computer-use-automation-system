"""Shared test builders for a valid CapabilityArtifact.

Centralized so schema tests, store tests, and (in later phases) replay
tests all exercise the same realistic `member_lookup_balance` shape rather
than each hand-rolling a slightly different fixture. Nothing here is
"discovered" data -- it is a hand-authored example of what discovery would
produce, used purely to exercise validation.
"""
from __future__ import annotations

from cua.actions import ActionType
from cua.artifact.checks import CheckpointSpec, UrlMatchesCheck
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
from cua.artifact.targets import TargetDescriptor


def member_id_target() -> TargetDescriptor:
    return TargetDescriptor(role="textbox", accessible_name="Member ID")


def valid_steps() -> list[Step]:
    return [
        Step(step_id="s1_navigate", action=ActionType.NAVIGATE, params={"route": "/search"}),
        Step(
            step_id="s2_fill",
            action=ActionType.FILL,
            target=member_id_target(),
            params={"value": "{{member_id}}"},
        ),
        Step(
            step_id="s3_click",
            action=ActionType.CLICK,
            target=TargetDescriptor(role="button", accessible_name="Search"),
        ),
        Step(
            step_id="s4_checkpoint",
            action=ActionType.CHECKPOINT,
            checkpoint=CheckpointSpec(
                description="results page reached", assertion=UrlMatchesCheck(pattern="/results")
            ),
        ),
        Step(
            step_id="s5_extract",
            action=ActionType.READ_EXTRACT,
            target=TargetDescriptor(role="cell", accessible_name="Savings Balance"),
            extraction=ExtractionSpec(
                output_name="savings_balance",
                target=TargetDescriptor(role="cell", accessible_name="Savings Balance"),
            ),
        ),
    ]


def base_kwargs(steps: list[Step] | None = None) -> dict:
    return dict(
        capability_id="member_lookup_balance",
        capability_version="1.0.0",
        name="Look up member and read savings balance",
        description="Search for a member by id and extract their current savings balance.",
        target=TargetIdentity(app_name="CU Ops Console", base_url_pattern="http://127.0.0.1:8000"),
        risk=RiskLevel.SAFE,
        inputs=[
            InputParam(
                name="member_id", type=ParamType.STRING, description="Member identifier", example="12345"
            )
        ],
        outputs=[OutputParam(name="savings_balance", type=ParamType.STRING, description="Current savings balance")],
        steps=steps if steps is not None else valid_steps(),
        created_by="discovery_agent",
    )


def build_artifact(**overrides) -> CapabilityArtifact:
    kwargs = base_kwargs()
    kwargs.update(overrides)
    return CapabilityArtifact(**kwargs)
