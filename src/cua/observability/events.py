"""Structured log event schema (Section 3.5)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class LogEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str
    mode: str  # "discovery" | "replay"
    capability_id: str | None = None
    capability_version: str | None = None
    step_id: str | None = None
    event_type: str  # e.g. "step_start", "step_end", "checkpoint", "policy_decision", "error"
    action: str | None = None
    target: str | None = None  # short human-readable description, never a raw observed value
    status: str | None = None
    duration_ms: int | None = None
    attempt: int = 1
    error_class: str | None = None
    evidence_reference: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
