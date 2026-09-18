import json
from pathlib import Path

from cua.observability.events import LogEvent
from cua.observability.logger import StructuredLogger


def test_log_writes_one_json_line(tmp_path: Path):
    logger = StructuredLogger(run_id="run123", log_dir=tmp_path)
    logger.log(LogEvent(run_id="run123", mode="discovery", event_type="step_start"))
    logger.log(LogEvent(run_id="run123", mode="discovery", event_type="step_end", status="SUCCESS"))
    lines = logger.path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["run_id"] == "run123"
    assert first["event_type"] == "step_start"


def test_log_redacts_secret_shaped_detail(tmp_path: Path):
    logger = StructuredLogger(run_id="run456", log_dir=tmp_path)
    logger.log(
        LogEvent(
            run_id="run456",
            mode="discovery",
            event_type="model_call",
            detail={"note": "using token sk-shouldnotappearinlogsxxxx"},
        )
    )
    raw = logger.path.read_text()
    assert "sk-shouldnotappearinlogsxxxx" not in raw


def test_log_file_named_by_run_id(tmp_path: Path):
    logger = StructuredLogger(run_id="abc-123", log_dir=tmp_path)
    assert logger.path == tmp_path / "abc-123.jsonl"
