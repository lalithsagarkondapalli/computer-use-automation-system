"""DiscoveryLoop: the genuine OBSERVE -> DECIDE -> VALIDATE -> POLICY CHECK
-> ACT -> OBSERVE cycle (Phase 4 objective).

This is the one place in the repository where an LLM's output can
influence what happens on a live surface -- and even here, the model only
ever returns a typed `AgentAction` (via `ModelAdapter.decide`); every
proposal still has to survive schema validation (inside `AgentAction`
itself), semantic VALIDATE (`cua.agent.validation`: is this grounded in
what was actually observed), and POLICY CHECK (`cua.safety.engine.PolicyEngine`,
the *same* engine `ReplayEngine` uses) before this loop ever calls a
`SurfaceAdapter` method. Nothing here executes arbitrary code, a shell
command, raw Playwright, or an unconstrained HTTP request -- the closed
`ActionType` vocabulary (`cua.actions`) is the entire surface the model
can touch.

Stopping conditions (Section 3.1) are all represented as a
`DiscoveryStopReason` on the returned `DiscoveryTrace` -- there is no path
through `run()` that loops forever or returns without one.
"""
from __future__ import annotations

import time
import uuid
from urllib.parse import urljoin

from cua.actions import ActionType
from cua.agent.actions import AgentAction
from cua.agent.checkpoint_synthesis import synthesize_checkpoint
from cua.agent.evidence import DiscoveryEvidenceCapture
from cua.agent.fingerprint import fingerprint_action, fingerprint_observation
from cua.agent.inference import infer_read_attribute
from cua.agent.risk import classify_action_risk
from cua.agent.trace import DiscoveryStepRecord, DiscoveryStopReason, DiscoveryTrace
from cua.agent.validation import validate_proposal
from cua.artifact.checks import CheckpointAssertion, CheckpointSpec
from cua.artifact.extraction import ExtractionSpec
from cua.artifact.schema import Step
from cua.models.base import ModelAdapter, ModelOutputError
from cua.observability.events import LogEvent
from cua.observability.logger import StructuredLogger
from cua.safety.engine import PolicyEngine
from cua.surfaces.base import SurfaceAdapter, TargetAmbiguous, TargetNotFound
from cua.surfaces.observation import Observation


class DiscoveryLoop:
    def __init__(
        self,
        model: ModelAdapter,
        surface: SurfaceAdapter,
        policy: PolicyEngine,
        logger: StructuredLogger,
        evidence: DiscoveryEvidenceCapture | None = None,
        *,
        max_steps: int = 25,
        overall_timeout_s: float = 300,
        stuck_repeat_threshold: int = 3,
        max_invalid_output_retries: int = 2,
        step_timeout_ms: int = 10_000,
    ):
        self._model = model
        self._surface = surface
        self._policy = policy
        self._logger = logger
        self._evidence = evidence
        self._max_steps = max_steps
        self._overall_timeout_s = overall_timeout_s
        self._stuck_repeat_threshold = stuck_repeat_threshold
        self._max_invalid_output_retries = max_invalid_output_retries
        self._step_timeout_ms = step_timeout_ms

    async def run(self, goal: str) -> DiscoveryTrace:
        run_id = str(uuid.uuid4())
        trace = DiscoveryTrace(run_id=run_id, goal=goal)
        started = time.monotonic()
        history: list[str] = [f"goal: {goal}"]
        last_pair_fp: str | None = None
        repeat_count = 0

        observation = await self._surface.observe(step_number=0)
        baseline_observation = observation

        for step_number in range(1, self._max_steps + 1):
            if time.monotonic() - started > self._overall_timeout_s:
                return await self._stop(trace, run_id, step_number, DiscoveryStopReason.TIMEOUT,
                                   f"discovery exceeded the overall timeout of {self._overall_timeout_s}s")

            record = DiscoveryStepRecord(step_number=step_number, observation_before=observation)
            self._log(run_id, step_number, "observe", detail={"url": observation.url, "title": observation.title})

            action, checkpoint_assertion, invalid_reason, attempts = await self._decide_validated(
                run_id, step_number, observation, baseline_observation, history
            )
            record.invalid_output_attempts = attempts

            if action is None:
                record.validation_error = invalid_reason
                trace.steps.append(record)
                return await self._stop(
                    trace, run_id, step_number, DiscoveryStopReason.INVALID_OUTPUT_EXHAUSTED,
                    f"exhausted {self._max_invalid_output_retries} correction attempts: {invalid_reason}",
                )
            record.action = action

            if action.action == ActionType.FINISH:
                record.executed = True
                trace.steps.append(record)
                return await self._stop(trace, run_id, step_number, DiscoveryStopReason.GOAL_ACHIEVED, action.reasoning)

            if action.action == ActionType.REQUEST_HUMAN:
                record.executed = True
                trace.steps.append(record)
                return await self._stop(
                    trace, run_id, step_number, DiscoveryStopReason.HUMAN_REQUESTED,
                    action.params.get("reason", "model requested human intervention"),
                )

            pair_fp = f"{fingerprint_observation(observation)}:{fingerprint_action(action)}"
            repeat_count = repeat_count + 1 if pair_fp == last_pair_fp else 1
            last_pair_fp = pair_fp
            if repeat_count >= self._stuck_repeat_threshold:
                trace.steps.append(record)
                return await self._stop(
                    trace, run_id, step_number, DiscoveryStopReason.STUCK_STATE,
                    f"the same action was proposed against the same observed state {repeat_count} times in a row",
                )

            # -- POLICY CHECK --
            if action.action == ActionType.NAVIGATE:
                target_url = self._resolve_navigation_url(
                    await self._surface.current_url(), action.params.get("url") or action.params.get("route", "")
                )
                nav_decision = self._policy.check_domain(target_url)
                if not nav_decision.allowed:
                    record.policy_decision = nav_decision
                    trace.steps.append(record)
                    self._log(run_id, step_number, "policy_reject", detail={"reason": nav_decision.reason})
                    return await self._stop(trace, run_id, step_number, DiscoveryStopReason.POLICY_BLOCKED, nav_decision.reason)

            policy_step = self._build_policy_step(action, checkpoint_assertion)
            action_decision = self._policy.check_action(policy_step)
            record.policy_decision = action_decision
            if not action_decision.allowed:
                trace.steps.append(record)
                self._log(run_id, step_number, "policy_reject", detail={"reason": action_decision.reason})
                return await self._stop(trace, run_id, step_number, DiscoveryStopReason.POLICY_BLOCKED, action_decision.reason)

            # -- ACT --
            try:
                extracted_value = await self._execute(action)
            except (TargetNotFound, TargetAmbiguous) as exc:
                record.execution_error = str(exc)
                trace.steps.append(record)
                self._log(run_id, step_number, "act_error", error_class=type(exc).__name__, detail={"error": str(exc)})
                history.append(f"step {step_number}: {action.action.value} failed -- {exc}")
                observation = await self._surface.observe(step_number=step_number)
                continue  # let the model see the failure on the next OBSERVE and decide what to do
            except Exception as exc:  # pragma: no cover -- exercised via a scripted surface in tests
                record.execution_error = f"unrecoverable surface error: {exc}"
                trace.steps.append(record)
                self._log(run_id, step_number, "surface_error", error_class=type(exc).__name__, detail={"error": str(exc)})
                return await self._stop(trace, run_id, step_number, DiscoveryStopReason.SURFACE_ERROR, str(exc))

            record.executed = True
            if action.action == ActionType.CHECKPOINT:
                record.checkpoint_assertion = checkpoint_assertion
            if action.action == ActionType.READ_EXTRACT:
                record.extracted_output_name = action.params.get("output_name")
                record.extracted_value = extracted_value

            if action.action in (ActionType.NAVIGATE, ActionType.CLICK):
                current = await self._surface.current_url()
                post_decision = self._policy.check_domain(current)
                if not post_decision.allowed:
                    record.policy_decision = post_decision
                    trace.steps.append(record)
                    self._log(run_id, step_number, "policy_reject", detail={"reason": post_decision.reason, "url": current})
                    return await self._stop(
                        trace, run_id, step_number, DiscoveryStopReason.POLICY_BLOCKED,
                        f"navigation escaped the allowlist: {post_decision.reason}",
                    )

            observation_after = await self._surface.observe(step_number=step_number)
            record.observation_after = observation_after
            trace.steps.append(record)
            self._log(run_id, step_number, "act", action=action.action.value, status="ok")
            history.append(self._history_line(step_number, action))
            if action.action != ActionType.CHECKPOINT:
                baseline_observation = observation
            observation = observation_after

        return await self._stop(trace, run_id, self._max_steps, DiscoveryStopReason.MAX_STEPS,
                           f"reached max_steps={self._max_steps} without FINISH")

    async def _decide_validated(
        self, run_id: str, step_number: int, observation: Observation, baseline_observation: Observation, history: list[str]
    ) -> tuple[AgentAction | None, CheckpointAssertion | None, str | None, int]:
        """DECIDE + VALIDATE, with bounded correction retry. Returns
        (action, checkpoint_assertion, last_rejection_reason, attempts_made).
        `action` is None iff every attempt was rejected."""
        last_reason: str | None = None
        for attempt in range(1, self._max_invalid_output_retries + 2):  # +1 initial +N retries
            try:
                candidate = await self._model.decide(observation, history)
            except ModelOutputError as exc:
                last_reason = str(exc)
                self._log(run_id, step_number, "decide_invalid", attempt=attempt, error_class="ModelOutputError", detail={"error": str(exc)})
                history.append(f"step {step_number} attempt {attempt}: your response was rejected -- {exc}. Propose a valid action.")
                continue

            self._log(run_id, step_number, "decide", attempt=attempt, action=candidate.action.value, detail={"reasoning": candidate.reasoning})

            checkpoint_assertion: CheckpointAssertion | None = None
            error = validate_proposal(candidate, observation)
            if error is None and candidate.action == ActionType.CHECKPOINT:
                checkpoint_assertion = synthesize_checkpoint(candidate.params.get("description", ""), baseline_observation, observation)
                if checkpoint_assertion is None:
                    error = "checkpoint claim does not match any verifiable state change since the previous action"

            if error is not None:
                last_reason = error
                self._log(run_id, step_number, "validate_reject", attempt=attempt, detail={"reason": error})
                history.append(f"step {step_number} attempt {attempt}: rejected -- {error}. Propose a different, valid action.")
                continue

            return candidate, checkpoint_assertion, None, attempt - 1

        return None, None, last_reason, self._max_invalid_output_retries

    async def _execute(self, action: AgentAction) -> str | None:
        if action.action == ActionType.NAVIGATE:
            await self._surface.navigate(action.params.get("url") or action.params.get("route", ""))
            return None
        if action.action == ActionType.CLICK:
            resolved = await self._surface.resolve(action.target)
            await self._surface.click(resolved)
            return None
        if action.action == ActionType.FILL:
            resolved = await self._surface.resolve(action.target)
            await self._surface.fill(resolved, action.params["value"])
            return None
        if action.action == ActionType.SELECT:
            resolved = await self._surface.resolve(action.target)
            await self._surface.select(resolved, action.params["value"])
            return None
        if action.action == ActionType.WAIT:
            ok = await self._surface.wait_for(action.target, self._step_timeout_ms)
            if not ok:
                raise TargetNotFound(f"wait timed out for target {action.target!r}")
            return None
        if action.action == ActionType.READ_EXTRACT:
            resolved = await self._surface.resolve(action.target)
            attribute = infer_read_attribute(action.target)
            return await self._surface.read(resolved, attribute)
        if action.action == ActionType.CHECKPOINT:
            return None  # nothing to execute; verified deterministically during VALIDATE
        raise AssertionError(f"unhandled action in discovery loop: {action.action}")  # pragma: no cover -- closed vocabulary

    def _build_policy_step(self, action: AgentAction, checkpoint_assertion: CheckpointAssertion | None) -> Step:
        risk = classify_action_risk(action)
        if action.action == ActionType.READ_EXTRACT:
            extraction = ExtractionSpec(
                output_name=action.params.get("output_name", "value"),
                target=action.target,
                attribute=infer_read_attribute(action.target),
            )
            return Step(step_id="proposal", action=action.action, target=action.target, extraction=extraction, risk=risk)
        if action.action == ActionType.CHECKPOINT:
            checkpoint = CheckpointSpec(description=action.params.get("description", ""), assertion=checkpoint_assertion)
            return Step(step_id="proposal", action=action.action, checkpoint=checkpoint, risk=risk)
        return Step(step_id="proposal", action=action.action, target=action.target, params=action.params, risk=risk)

    def _resolve_navigation_url(self, current_url: str, url_or_route: str) -> str:
        if url_or_route.startswith("http://") or url_or_route.startswith("https://"):
            return url_or_route
        return urljoin(current_url, url_or_route)

    def _history_line(self, step_number: int, action: AgentAction) -> str:
        target_desc = ""
        if action.target is not None:
            target_desc = f" on {action.target.accessible_name or action.target.nearby_anchor or action.target.visible_text or action.target.dom_selector}"
        return f"step {step_number}: {action.action.value}{target_desc} -> ok"

    async def _stop(self, trace: DiscoveryTrace, run_id: str, step_number: int, reason: DiscoveryStopReason, message: str) -> DiscoveryTrace:
        trace.stop_reason = reason
        trace.stop_message = message
        # Evidence capture mirrors ReplayEngine's: taken on every stop that
        # isn't a clean success, since GOAL_ACHIEVED needs no debugging aid.
        if self._evidence is not None and reason != DiscoveryStopReason.GOAL_ACHIEVED:
            ref = await self._evidence.capture(run_id, self._surface)
            if ref is not None:
                trace.evidence.append(ref)
        self._log(run_id, step_number, "stop", status=reason.value, detail={"message": message})
        return trace

    def _log(self, run_id: str, step_number: int | None, event_type: str, **kwargs: object) -> None:
        detail = kwargs.pop("detail", {})
        self._logger.log(
            LogEvent(
                run_id=run_id,
                mode="discovery",
                step_id=str(step_number) if step_number is not None else None,
                event_type=event_type,
                detail=detail,
                **{k: v for k, v in kwargs.items() if k in {"action", "status", "attempt", "error_class"}},
            )
        )
