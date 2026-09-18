"""Redaction of secret-shaped values before they reach logs or artifacts.

Section 3.4: "never persist secrets or raw sensitive data ... redact
appropriately." This is a deliberately conservative heuristic, not a claim
of perfect PII detection -- it catches the specific shapes that matter for
this system (API keys/tokens, bearer headers, card-number-length digit
runs, SSN-shaped values) and is applied uniformly at the two points data
can leave the process: `cua.observability.logger.StructuredLogger` and
`cua.artifact.store` inputs (artifacts never contain observed runtime
values at all -- see `Step.params` interpolation -- so this is a second,
independent layer, not the only one).
"""
from __future__ import annotations

import re
from typing import Any

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(sk-|api[_-]?key\S*[:=]\s*|bearer\s+)\S{6,}"),
    re.compile(r"\b\d{13,19}\b"),  # card-number-length digit runs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped
)

REDACTED = "[REDACTED]"


def redact_text(value: str) -> str:
    out = value
    for pattern in _PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def redact_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact string values in a nested dict. Non-string,
    non-dict values (numbers, bools, None) pass through unchanged."""
    result: dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, str):
            result[key] = redact_text(value)
        elif isinstance(value, dict):
            result[key] = redact_detail(value)
        else:
            result[key] = value
    return result
