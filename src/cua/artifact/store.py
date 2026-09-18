"""Persist and load a CapabilityArtifact as human-reviewable YAML.

YAML, not a database, is the deliberate choice: Section 3.2 requires the
artifact to be reviewable by a human, and a file a reviewer can open in a
text editor beats a row in a table for that purpose. Loading always goes
through the same Pydantic validation as construction, so a hand-edited or
drifted file on disk fails exactly the same way a bad discovery output
would -- there is no separate, weaker "load path."
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from cua.artifact.schema import CapabilityArtifact


class ArtifactLoadError(Exception):
    """Raised when a file on disk is not a valid, current-schema CapabilityArtifact."""


def save_artifact(artifact: CapabilityArtifact, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact.capability_id}@{artifact.capability_version}.yaml"
    data = artifact.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def load_artifact(path: Path) -> CapabilityArtifact:
    if not path.exists():
        raise ArtifactLoadError(f"no artifact at {path}")
    raw = yaml.safe_load(path.read_text())
    try:
        return CapabilityArtifact.model_validate(raw)
    except ValidationError as exc:
        raise ArtifactLoadError(f"{path} is not a valid capability artifact:\n{exc}") from exc
