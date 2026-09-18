"""Checkpoint verification: turn a CheckpointSpec's assertion (and any
declared condition probes) into a classified outcome, never a bare bool.

A checkpoint's job is exactly the assignment glossary's definition: "a
condition you assert to confirm you actually reached the state you
expected, rather than assuming a click worked." Probes are checked first,
each with its own short timeout, so a known alternate condition (a
"not found" message, an unexpected modal) is recognized before the main
assertion's full timeout is spent waiting for something that was never
going to happen -- this is the mechanism that keeps CUSTOMER_NOT_FOUND
from ever surfacing as a generic CHECKPOINT_FAILED.
"""
from __future__ import annotations

from dataclasses import dataclass

from cua.artifact.checks import CheckpointSpec
from cua.results.taxonomy import OutcomeCode, RunStatus
from cua.surfaces.base import SurfaceAdapter


@dataclass(frozen=True)
class CheckpointOutcome:
    passed: bool
    status: RunStatus = RunStatus.SUCCESS  # meaningful only when passed is False
    code: OutcomeCode | None = None
    observed: str = ""


class CheckpointEngine:
    async def verify(self, surface: SurfaceAdapter, checkpoint: CheckpointSpec, timeout_ms: int) -> CheckpointOutcome:
        for probe in checkpoint.probes:
            hit = await surface.wait_for_checkpoint(probe.assertion, probe.probe_timeout_ms)
            if hit:
                return CheckpointOutcome(passed=False, status=probe.status, code=probe.code, observed=probe.description)

        passed = await surface.wait_for_checkpoint(checkpoint.assertion, timeout_ms)
        if passed:
            return CheckpointOutcome(passed=True)
        return CheckpointOutcome(
            passed=False,
            status=RunStatus.HARD_FAILURE,
            code=OutcomeCode.CHECKPOINT_FAILED,
            observed=f"checkpoint not satisfied: {checkpoint.description}",
        )
