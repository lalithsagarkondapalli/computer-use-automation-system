"""ReplayEngine: deterministic execution of a CapabilityArtifact.

Zero LLM decisions happen here, or anywhere this module transitively
imports -- see tests/test_replay_model_isolation.py for the automated
proof. Every branch below is plain, typed Python: interpolate parameters,
check policy, act on the surface, verify a checkpoint, extract outputs,
classify the result.

Pipeline (Section 9): artifact load/validation (done by the caller before
`run()` -- see `cua.artifact.store.load_artifact`) -> invocation input
validation -> parameter interpolation -> policy validation -> surface/
session setup (also the caller's job -- see the module docstring note
below) -> deterministic step execution -> locator resolution -> action ->
checkpoint/postcondition -> extraction -> structured result.

Session lifecycle is deliberately NOT owned by this class: the caller
constructs and opens a `SurfaceAdapter` (e.g. via
`PlaywrightWebSurface.launch()`) and is responsible for closing it. This
keeps `ReplayEngine` trivially unit-testable against a `FakeSurfaceAdapter`
that never needs a real browser, and keeps "how a session is opened" a
concern of the caller (CLI, test, or future handoff console) rather than
baked into the engine.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from cua.actions import ActionType
from cua.artifact.schema import CapabilityArtifact, Step
from cua.observability.events import LogEvent
from cua.observability.logger import StructuredLogger
from cua.replay.checkpoints import CheckpointEngine
from cua.replay.evidence import EvidenceCapture
from cua.replay.extraction import extract_output
from cua.replay.interpolation import InputValidationError, interpolate_steps, validate_inputs
from cua.results.taxonomy import OutcomeCode, RecoveryAttempt, RunResult, RunStatus
from cua.safety.engine import PolicyEngine
from cua.surfaces.base import SurfaceAdapter, TargetAmbiguous, TargetNotFound


class ReplayEngine:
    def __init__(self, policy: PolicyEngine, logger: StructuredLogger, evidence: EvidenceCapture):
        self._policy = policy
        self._logger = logger
        self._evidence = evidence

    async def run(self, artifact: CapabilityArtifact, raw_inputs: dict[str, Any], surface: SurfaceAdapter) -> RunResult:
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        recovery_attempts: list[RecoveryAttempt] = []

        # -- invocation input validation --------------------------------
        try:
            inputs = validate_inputs(artifact, raw_inputs)
        except InputValidationError as exc:
            return await self._fail(
                run_id, artifact, started_at, surface, recovery_attempts,
                RunStatus.HARD_FAILURE, OutcomeCode.INVALID_INPUT, message=str(exc),
            )

        # -- parameter interpolation -------------------------------------
        try:
            steps = interpolate_steps(artifact.steps, inputs)
        except InputValidationError as exc:
            return await self._fail(
                run_id, artifact, started_at, surface, recovery_attempts,
                RunStatus.HARD_FAILURE, OutcomeCode.INVALID_INPUT, message=str(exc),
            )

        # -- upfront policy validation against the capability's own target
        domain_decision = self._policy.check_domain(artifact.target.base_url_pattern)
        if not domain_decision.allowed:
            return await self._fail(
                run_id, artifact, started_at, surface, recovery_attempts,
                RunStatus.POLICY_BLOCKED, OutcomeCode.DOMAIN_NOT_ALLOWED, message=domain_decision.reason,
            )

        outputs: dict[str, Any] = {}

        for step in steps:
            step_started = time.monotonic()
            self._log(run_id, artifact, step.step_id, "step_start", action=step.action.value)

            # -- policy check BEFORE the action reaches the surface -------
            action_decision = self._policy.check_action(step)
            if not action_decision.allowed:
                code = (
                    OutcomeCode.RISKY_ACTION_REQUIRES_APPROVAL
                    if action_decision.requires_approval
                    else OutcomeCode.ACTION_TYPE_NOT_ALLOWED
                )
                return await self._fail(
                    run_id, artifact, started_at, surface, recovery_attempts,
                    RunStatus.POLICY_BLOCKED, code, message=action_decision.reason, failing_step_id=step.step_id,
                )

            if step.action == ActionType.NAVIGATE:
                target_url = self._resolve_navigation_url(artifact, step.params.get("url") or step.params.get("route", ""))
                nav_decision = self._policy.check_domain(target_url)
                if not nav_decision.allowed:
                    return await self._fail(
                        run_id, artifact, started_at, surface, recovery_attempts,
                        RunStatus.POLICY_BLOCKED, OutcomeCode.DOMAIN_NOT_ALLOWED,
                        message=nav_decision.reason, failing_step_id=step.step_id,
                    )

            # -- action, with bounded retry --------------------------------
            try:
                await self._execute_with_retry(surface, step, outputs, recovery_attempts)
            except TargetAmbiguous as exc:
                return await self._fail(
                    run_id, artifact, started_at, surface, recovery_attempts,
                    RunStatus.ESCALATION_REQUIRED, OutcomeCode.AMBIGUOUS_CONTROL,
                    message=str(exc), failing_step_id=step.step_id,
                    expected_state="exactly one control matched", observed_state=f"{exc.count} candidates matched via {exc.strategy}",
                )
            except TargetNotFound as exc:
                return await self._fail(
                    run_id, artifact, started_at, surface, recovery_attempts,
                    RunStatus.HARD_FAILURE, OutcomeCode.TARGET_UNRESOLVABLE,
                    message=str(exc), failing_step_id=step.step_id,
                )

            # -- post-action domain-escape check (Section 3.4: redirects) --
            if step.action in (ActionType.NAVIGATE, ActionType.CLICK):
                current = await surface.current_url()
                post_decision = self._policy.check_domain(current)
                if not post_decision.allowed:
                    return await self._fail(
                        run_id, artifact, started_at, surface, recovery_attempts,
                        RunStatus.POLICY_BLOCKED, OutcomeCode.DOMAIN_NOT_ALLOWED,
                        message=f"navigation escaped the allowlist: {post_decision.reason}",
                        failing_step_id=step.step_id, observed_state=current,
                    )

            # -- checkpoint verification (own outcome, not bool/exception) -
            if step.action == ActionType.CHECKPOINT:
                outcome = await CheckpointEngine().verify(surface, step.checkpoint, step.timeout_ms)
                if not outcome.passed:
                    return await self._fail(
                        run_id, artifact, started_at, surface, recovery_attempts,
                        outcome.status, outcome.code,
                        message=outcome.observed, failing_step_id=step.step_id,
                        expected_state=step.checkpoint.description, observed_state=outcome.observed,
                    )

            duration_ms = int((time.monotonic() - step_started) * 1000)
            self._log(run_id, artifact, step.step_id, "step_end", action=step.action.value, status="ok", duration_ms=duration_ms)

        return RunResult(
            run_id=run_id,
            mode="replay",
            status=RunStatus.SUCCESS,
            capability_id=artifact.capability_id,
            capability_version=artifact.capability_version,
            outputs=outputs,
            recovery_attempts=recovery_attempts,
            message="completed successfully",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    async def _execute_with_retry(
        self, surface: SurfaceAdapter, step: Step, outputs: dict[str, Any], recovery_attempts: list[RecoveryAttempt]
    ) -> None:
        last_exc: TargetNotFound | None = None
        for attempt in range(1, step.retry.max_attempts + 1):
            try:
                await self._execute_once(surface, step, outputs)
                return
            except TargetAmbiguous:
                # Never retry an ambiguity: retrying can't resolve a genuine
                # multi-match, and doing so risks masking it if a later
                # attempt "gets lucky" against a slightly different DOM state.
                raise
            except TargetNotFound as exc:
                last_exc = exc
                if attempt < step.retry.max_attempts:
                    recovery_attempts.append(
                        RecoveryAttempt(step_id=step.step_id, strategy="retry_target_not_found", outcome="failed", detail=f"attempt {attempt}")
                    )
                    await asyncio.sleep(step.retry.backoff_ms / 1000)
                    continue
                raise
        if last_exc is not None:  # pragma: no cover -- loop above always returns or raises
            raise last_exc

    async def _execute_once(self, surface: SurfaceAdapter, step: Step, outputs: dict[str, Any]) -> None:
        if step.action == ActionType.NAVIGATE:
            await surface.navigate(step.params.get("url") or step.params.get("route", ""))
        elif step.action == ActionType.CLICK:
            resolved = await surface.resolve(step.target)
            await surface.click(resolved)
        elif step.action == ActionType.FILL:
            resolved = await surface.resolve(step.target)
            await surface.fill(resolved, step.params["value"])
        elif step.action == ActionType.SELECT:
            resolved = await surface.resolve(step.target)
            await surface.select(resolved, step.params["value"])
        elif step.action == ActionType.WAIT:
            ok = await surface.wait_for(step.target, step.timeout_ms)
            if not ok:
                raise TargetNotFound(f"wait timed out for target in step {step.step_id}")
        elif step.action == ActionType.READ_EXTRACT:
            outputs[step.extraction.output_name] = await extract_output(surface, step.extraction)
        elif step.action == ActionType.CHECKPOINT:
            pass  # handled by the caller in run(), which needs the richer CheckpointOutcome
        else:  # pragma: no cover -- Step validation restricts action to STEP_ACTIONS
            raise AssertionError(f"unhandled action type in replay: {step.action}")

    def _resolve_navigation_url(self, artifact: CapabilityArtifact, url_or_route: str) -> str:
        if url_or_route.startswith("http://") or url_or_route.startswith("https://"):
            return url_or_route
        return f"{artifact.target.base_url_pattern.rstrip('/')}{url_or_route}"

    async def _fail(
        self,
        run_id: str,
        artifact: CapabilityArtifact,
        started_at: datetime,
        surface: SurfaceAdapter,
        recovery_attempts: list[RecoveryAttempt],
        status: RunStatus,
        code: OutcomeCode | None,
        *,
        message: str,
        failing_step_id: str | None = None,
        expected_state: str | None = None,
        observed_state: str | None = None,
    ) -> RunResult:
        evidence_refs = []
        ref = await self._evidence.capture(run_id, surface)
        if ref is not None:
            evidence_refs.append(ref)
        self._log(run_id, artifact, failing_step_id, "error", status=status.value, error_class=code.value if code else None)
        return RunResult(
            run_id=run_id,
            mode="replay",
            status=status,
            code=code,
            capability_id=artifact.capability_id,
            capability_version=artifact.capability_version,
            failing_step_id=failing_step_id,
            expected_state=expected_state,
            observed_state=observed_state,
            recovery_attempts=recovery_attempts,
            evidence=evidence_refs,
            message=message,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    def _log(self, run_id: str, artifact: CapabilityArtifact, step_id: str | None, event_type: str, **kwargs: Any) -> None:
        self._logger.log(
            LogEvent(
                run_id=run_id,
                mode="replay",
                capability_id=artifact.capability_id,
                capability_version=artifact.capability_version,
                step_id=step_id,
                event_type=event_type,
                **kwargs,
            )
        )
