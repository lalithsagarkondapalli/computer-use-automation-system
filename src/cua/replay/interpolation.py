"""Parameter interpolation and invocation input validation.

Section 9's replay pipeline is: artifact load/validation -> invocation
input validation -> parameter interpolation -> policy validation -> ...
This module is the second and third stages. Validation happens once,
up front, against a copy of the artifact's steps; by the time step
execution starts, every param value is a concrete string and nothing
later in the pipeline needs to know `{{...}}` syntax exists.
"""
from __future__ import annotations

import re
from typing import Any

from cua.artifact.schema import CapabilityArtifact, Step

_TOKEN = re.compile(r"\{\{(\w+)\}\}")


class InputValidationError(Exception):
    """A caller-supplied invocation parameter is missing, unexpected, or the wrong type.

    Maps to RunStatus.HARD_FAILURE / OutcomeCode.INVALID_INPUT in
    ReplayEngine -- this is a caller/integration contract violation, not a
    business outcome from the target application and not something a
    retry or a human could fix.
    """


def validate_inputs(artifact: CapabilityArtifact, raw_inputs: dict[str, Any]) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for param in artifact.inputs:
        if param.name not in raw_inputs:
            if param.required:
                raise InputValidationError(f"missing required input '{param.name}'")
            continue
        try:
            validated[param.name] = _coerce(raw_inputs[param.name], param.type.value)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"input '{param.name}' must be {param.type.value}: {exc}") from exc

    unexpected = set(raw_inputs) - {p.name for p in artifact.inputs}
    if unexpected:
        raise InputValidationError(f"unexpected inputs not declared by this capability: {sorted(unexpected)}")
    return validated


def _coerce(value: Any, type_name: str) -> Any:
    if type_name == "string":
        return str(value)
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"cannot coerce {value!r} to bool")
    raise ValueError(f"unknown param type {type_name!r}")  # pragma: no cover -- ParamType is a closed enum


def interpolate_steps(steps: list[Step], inputs: dict[str, Any]) -> list[Step]:
    """Return a new list of steps with every `{{token}}` in `params`
    replaced by the corresponding validated input value. Does not mutate
    the artifact's own steps -- the same artifact can be replayed
    concurrently with different inputs without steps from one run leaking
    into another.
    """
    rendered: list[Step] = []
    for step in steps:
        try:
            new_params = {key: _TOKEN.sub(lambda m: str(inputs[m.group(1)]), value) for key, value in step.params.items()}
        except KeyError as exc:
            raise InputValidationError(
                f"step {step.step_id} references input {exc} which was not supplied"
            ) from exc
        rendered.append(step.model_copy(update={"params": new_params}))
    return rendered
