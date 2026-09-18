"""ArtifactBuilder: turns a successful `DiscoveryTrace` into the same typed,
versioned `CapabilityArtifact` the Phase 3 `ReplayEngine` already consumes
(Section 6 -- this is the critical discovery/replay boundary).

Deliberately NOT "serialize the transcript" -- a raw discovery run is full
of exactly the things a persisted capability must not contain: literal
runtime values (a specific member id), discovery-only control actions
(FINISH, REQUEST_HUMAN), and steps that failed and were retried. This
module is the one place that decides what from a trace is durable
capability knowledge versus one run's incidental detail, and it does so
by re-deriving every artifact field from structured trace data -- never
by asking the model to describe the capability after the fact, which
would reopen exactly the "trust the model's self-report" problem the rest
of this system is built to avoid.
"""
from __future__ import annotations

from pydantic import ValidationError

from cua.actions import ActionType
from cua.agent.checkpoint_synthesis import synthesize_checkpoint
from cua.agent.inference import infer_read_attribute, infer_transform
from cua.agent.risk import classify_action_risk
from cua.agent.trace import DiscoveryStepRecord, DiscoveryStopReason, DiscoveryTrace
from cua.artifact.checks import CheckpointSpec, ConditionProbe, ElementVisibleCheck
from cua.artifact.extraction import ExtractionSpec
from cua.artifact.schema import CapabilityArtifact, InputParam, OutputParam, RiskLevel, Step, TargetIdentity
from cua.safety.redaction import redact_text

_RISK_ORDER = (RiskLevel.SAFE, RiskLevel.SENSITIVE, RiskLevel.IRREVERSIBLE)
_DISCOVERY_ONLY = frozenset({ActionType.FINISH, ActionType.REQUEST_HUMAN})


class ArtifactBuildError(Exception):
    """Raised when a trace cannot be turned into a valid CapabilityArtifact
    -- either because the trace itself doesn't represent a completed,
    successful run, or because the resulting artifact fails its own
    schema validation. Never silently produces a partial/best-effort
    artifact."""


class ArtifactBuilder:
    def build(
        self,
        trace: DiscoveryTrace,
        *,
        capability_id: str,
        capability_version: str,
        name: str,
        description: str,
        app_name: str,
        base_url_pattern: str,
        inputs: list[InputParam],
        outputs: list[OutputParam],
        parameter_values: dict[str, str] | None = None,
        known_business_outcome_probes: list[ConditionProbe] | None = None,
        created_by: str = "discovery_agent",
    ) -> CapabilityArtifact:
        if trace.stop_reason != DiscoveryStopReason.GOAL_ACHIEVED:
            raise ArtifactBuildError(
                f"refusing to build an artifact from a trace that did not reach GOAL_ACHIEVED "
                f"(stop_reason={trace.stop_reason})"
            )

        parameter_values = parameter_values or {}
        declared_outputs = {o.name for o in outputs}
        executed = [
            record
            for record in trace.steps
            if record.executed and record.action is not None and record.action.action not in _DISCOVERY_ONLY
        ]
        if not executed:
            raise ArtifactBuildError("trace has no executed, persistable steps -- nothing to build a capability from")

        steps: list[Step] = []
        risk_seen: set[RiskLevel] = set()
        for index, record in enumerate(executed, start=1):
            step = self._build_step(index, record, parameter_values, declared_outputs)
            risk_seen.add(step.risk)
            steps.append(step)

        if not any(s.action == ActionType.CHECKPOINT for s in steps):
            # The model reached GOAL_ACHIEVED without ever explicitly proposing
            # a CHECKPOINT action -- nothing in the action schema or prompt
            # requires it to. Rather than accept a bare FINISH as proof of
            # success (which would let an ungrounded claim into a persisted
            # capability), deterministically re-derive a checkpoint from the
            # trace's own structured data, the same way every other field in
            # this builder is re-derived rather than trusted from the model.
            checkpoint_step, insert_at = self._synthesize_missing_checkpoint(executed)
            steps.insert(insert_at, checkpoint_step)
            risk_seen.add(checkpoint_step.risk)

        if known_business_outcome_probes:
            self._attach_probes_to_last_checkpoint(steps, known_business_outcome_probes)

        capability_risk = max(risk_seen, key=_RISK_ORDER.index) if risk_seen else RiskLevel.SAFE

        try:
            return CapabilityArtifact(
                capability_id=capability_id,
                capability_version=capability_version,
                name=name,
                description=description,
                target=TargetIdentity(app_name=app_name, base_url_pattern=base_url_pattern),
                risk=capability_risk,
                inputs=inputs,
                outputs=outputs,
                steps=steps,
                created_by=created_by,
                discovery_run_id=trace.run_id,
            )
        except ValidationError as exc:
            raise ArtifactBuildError(f"discovery trace produced an invalid artifact:\n{exc}") from exc

    def _build_step(
        self, index: int, record: DiscoveryStepRecord, parameter_values: dict[str, str], declared_outputs: set[str]
    ) -> Step:
        action = record.action
        assert action is not None  # narrowed by the caller's filter
        step_id = f"s{index}_{action.action.value}"
        risk = classify_action_risk(action)

        if action.action == ActionType.READ_EXTRACT:
            output_name = action.params.get("output_name", "")
            if output_name not in declared_outputs:
                raise ArtifactBuildError(
                    f"step {step_id} extracts output {output_name!r}, which was not declared in `outputs`"
                )
            raw_value = record.extracted_value or ""
            extraction = ExtractionSpec(
                output_name=output_name,
                target=action.target,
                attribute=infer_read_attribute(action.target),
                transform=infer_transform(raw_value),
            )
            return Step(step_id=step_id, action=action.action, target=action.target, extraction=extraction, risk=risk)

        if action.action == ActionType.CHECKPOINT:
            if record.checkpoint_assertion is None:  # pragma: no cover -- the loop never executes an unsynthesized checkpoint
                raise ArtifactBuildError(f"step {step_id} is a checkpoint with no synthesized assertion")
            checkpoint = CheckpointSpec(
                description=action.params.get("description", ""), assertion=record.checkpoint_assertion
            )
            return Step(step_id=step_id, action=action.action, checkpoint=checkpoint, risk=risk)

        params = {key: self._parameterize(value, parameter_values) for key, value in action.params.items()}
        return Step(step_id=step_id, action=action.action, target=action.target, params=params, risk=risk)

    def _parameterize(self, value: str, parameter_values: dict[str, str]) -> str:
        """Replace a literal runtime value discovery actually used with the
        declared input's interpolation token -- e.g. the literal "12345"
        becomes "{{member_id}}" -- so the artifact never bakes in one run's
        specific data (Section 6: "parameterize runtime inputs", and
        Section 3.2's ground rule against hardcoded test data)."""
        result = value
        for param_name, literal in parameter_values.items():
            if literal and literal in result:
                result = result.replace(literal, f"{{{{{param_name}}}}}")
        return redact_text(result)

    def _synthesize_missing_checkpoint(self, executed: list[DiscoveryStepRecord]) -> tuple[Step, int]:
        """Deterministically re-derive a checkpoint from the trace's own
        structured data when the model reached GOAL_ACHIEVED without ever
        proposing a CHECKPOINT action itself.

        This never fabricates an observation the trace doesn't contain and
        never treats a bare FINISH as proof of success. Two priorities, both
        grounded in data the discovery surface actually returned:

        1. If the last executed READ_EXTRACT produced a real value from a
           concrete target, that target's visibility *is* the grounding --
           the extraction could not have produced a value unless the live
           surface resolved that exact element. Synthesize an
           ElementVisibleCheck on it.
        2. Otherwise, fall back to the same before/after observation diff
           `synthesize_checkpoint` already uses when the model itself
           proposes a CHECKPOINT (Phase 4), applied to the final executed
           step's own recorded observations.

        If neither yields a groundable assertion, refuse to build an
        artifact at all -- this is intentionally not a fallback that
        invents a generic, unverifiable checkpoint.

        Returns the new checkpoint `Step` plus the index at which it should
        be inserted into the artifact's `steps` list (indices line up 1:1
        with `executed` since both are built in trace order)."""
        for i in range(len(executed) - 1, -1, -1):
            record = executed[i]
            action = record.action
            if (
                action is not None
                and action.action == ActionType.READ_EXTRACT
                and record.extracted_value
                and action.target is not None
            ):
                output_name = action.params.get("output_name", "")
                checkpoint = CheckpointSpec(
                    description=(
                        f"the element read for output '{output_name}' was visible "
                        "(synthesized from a successful extraction; the model did "
                        "not explicitly emit a CHECKPOINT action)"
                    ),
                    assertion=ElementVisibleCheck(target=action.target),
                )
                step = Step(
                    step_id="checkpoint_synthesized_from_read_extract",
                    action=ActionType.CHECKPOINT,
                    checkpoint=checkpoint,
                    risk=RiskLevel.SAFE,
                )
                return step, i

        last = executed[-1]
        if last.observation_before is not None and last.observation_after is not None:
            assertion = synthesize_checkpoint("", last.observation_before, last.observation_after)
            if assertion is not None:
                checkpoint = CheckpointSpec(
                    description=(
                        "observable state changed across the final discovery step "
                        "(synthesized from a before/after observation diff; the "
                        "model did not explicitly emit a CHECKPOINT action)"
                    ),
                    assertion=assertion,
                )
                step = Step(
                    step_id="checkpoint_synthesized_from_observation_diff",
                    action=ActionType.CHECKPOINT,
                    checkpoint=checkpoint,
                    risk=RiskLevel.SAFE,
                )
                return step, len(executed)

        raise ArtifactBuildError(
            "trace reached GOAL_ACHIEVED but contains no groundable evidence to "
            "synthesize a checkpoint from (no successful READ_EXTRACT with a "
            "concrete target, and no observable state change on the final "
            "step) -- refusing to build an artifact on a bare FINISH claim"
        )

    def _attach_probes_to_last_checkpoint(self, steps: list[Step], probes: list[ConditionProbe]) -> None:
        for step in reversed(steps):
            if step.action == ActionType.CHECKPOINT and step.checkpoint is not None:
                step.checkpoint.probes = list(step.checkpoint.probes) + list(probes)
                return
        raise ArtifactBuildError("known_business_outcome_probes were supplied but the trace has no checkpoint step to attach them to")
