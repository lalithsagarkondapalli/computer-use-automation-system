"""JSONL structured logger.

One file per run (`<log_dir>/<run_id>.jsonl`), append-only, one JSON object
per line -- easy to `tail -f` while a run is in progress and easy to grep
across runs afterward (e.g. `mode: replay` lines with zero occurrences of
any model-adapter event_type is exactly the evidence that replay made no
LLM decisions). Every event's `detail` is passed through
`cua.safety.redaction` before it is written, so a secret- or PII-shaped
value can't reach disk even if a caller forgets to redact it upstream.
"""
from __future__ import annotations

from pathlib import Path

from cua.observability.events import LogEvent
from cua.safety.redaction import redact_detail


class StructuredLogger:
    def __init__(self, run_id: str, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{run_id}.jsonl"

    def log(self, event: LogEvent) -> None:
        safe_event = event.model_copy(update={"detail": redact_detail(event.detail)})
        with self._path.open("a") as f:
            f.write(safe_event.model_dump_json() + "\n")

    @property
    def path(self) -> Path:
        return self._path
