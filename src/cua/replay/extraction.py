"""Typed output extraction: read a value off the surface and apply a
named, closed transform (Section 3.2's "typed outputs / data to extract").

`transform` stays a closed set of reviewed operations, same rationale as
`CheckpointAssertion`: a human reviewer can read exactly what an artifact
will extract and how, without executing anything.
"""
from __future__ import annotations

from cua.artifact.extraction import ExtractionSpec
from cua.surfaces.base import SurfaceAdapter


class ExtractionError(Exception):
    """A resolved value could not be converted by the declared transform."""


async def extract_output(surface: SurfaceAdapter, spec: ExtractionSpec) -> str | float:
    """May raise TargetNotFound / TargetAmbiguous (propagated from
    `surface.resolve`) -- extraction failures are classified by the caller
    (ReplayEngine) the same way a click or fill's resolution failure is.
    """
    resolved = await surface.resolve(spec.target)
    raw = await surface.read(resolved, spec.attribute)
    return _apply_transform(raw, spec.transform)


def _apply_transform(raw: str, transform: str) -> str | float:
    if transform == "none":
        return raw
    if transform == "strip":
        return raw.strip()
    if transform == "currency_to_float":
        cleaned = raw.strip().lstrip("$").replace(",", "")
        try:
            return float(cleaned)
        except ValueError as exc:
            raise ExtractionError(f"cannot convert {raw!r} to a currency float") from exc
    raise ExtractionError(f"unknown transform {transform!r}")  # pragma: no cover -- closed enum on ExtractionSpec
