from pathlib import Path

import pytest

from cua.artifact.store import ArtifactLoadError, load_artifact, save_artifact

from factories import build_artifact


def test_save_and_load_round_trip(tmp_path: Path):
    artifact = build_artifact()
    path = save_artifact(artifact, tmp_path)
    assert path.exists()
    loaded = load_artifact(path)
    assert loaded == artifact


def test_saved_file_is_named_by_id_and_version(tmp_path: Path):
    artifact = build_artifact()
    path = save_artifact(artifact, tmp_path)
    assert path.name == "member_lookup_balance@1.0.0.yaml"


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(ArtifactLoadError):
        load_artifact(tmp_path / "nope.yaml")


def test_load_malformed_artifact_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("capability_id: only_one_field\n")
    with pytest.raises(ArtifactLoadError):
        load_artifact(bad)


def test_load_unsupported_schema_version_raises(tmp_path: Path):
    artifact = build_artifact()
    path = save_artifact(artifact, tmp_path)
    text = path.read_text().replace("schema_version: '1.0'", "schema_version: '2.0'")
    path.write_text(text)
    with pytest.raises(ArtifactLoadError):
        load_artifact(path)
