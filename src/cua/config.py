"""Application configuration.

Runtime configuration is entirely environment-driven so the system can be
run without editing code and without committing secrets (Section 3.4 /
Section 9 ground rules). This module deliberately has no dependency on
Playwright, FastAPI, or the model-provider SDK: it is pure config parsing
and validation, safe to import from anywhere -- including every test file
-- without pulling in a browser or a network client.

`anthropic_api_key` is read but marked `repr=False` so it can never leak
through a stray `print(config)` or an unguarded log line; see
`cua.safety.redaction` for the belt-and-suspenders check applied to
anything that actually reaches a log file or artifact.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Validated runtime configuration, loaded from environment variables."""

    anthropic_api_key: str | None = Field(default=None, repr=False)
    model_name: str = ""  # must be set via MODEL_NAME env var before running discovery

    demo_app_base_url: str = "http://127.0.0.1:8000"
    allowed_domains: tuple[str, ...] = ("127.0.0.1", "localhost")

    headed: bool = False

    max_discovery_steps: int = 25
    step_timeout_ms: int = 10_000

    artifact_dir: Path = Path("artifacts")
    evidence_dir: Path = Path("evidence")
    log_dir: Path = Path("logs")

    log_level: str = "INFO"

    @field_validator("allowed_domains", mode="before")
    @classmethod
    def _split_domains(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(d.strip() for d in v.split(",") if d.strip())
        return tuple(v)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from process environment variables.

        Entry points that should read a local `.env` file are responsible
        for calling `dotenv.load_dotenv()` before this; library and test
        code must not depend on `.env` being loaded implicitly.
        """
        kwargs: dict[str, object] = {}
        if (v := os.environ.get("ANTHROPIC_API_KEY")) is not None:
            kwargs["anthropic_api_key"] = v
        if (v := os.environ.get("MODEL_NAME")) is not None:
            kwargs["model_name"] = v
        if (v := os.environ.get("DEMO_APP_BASE_URL")) is not None:
            kwargs["demo_app_base_url"] = v
        if (v := os.environ.get("ALLOWED_DOMAINS")) is not None:
            kwargs["allowed_domains"] = v
        if (v := os.environ.get("HEADED")) is not None:
            kwargs["headed"] = v.strip().lower() in {"1", "true", "yes"}
        if (v := os.environ.get("MAX_DISCOVERY_STEPS")) is not None:
            kwargs["max_discovery_steps"] = int(v)
        if (v := os.environ.get("STEP_TIMEOUT_MS")) is not None:
            kwargs["step_timeout_ms"] = int(v)
        if (v := os.environ.get("ARTIFACT_DIR")) is not None:
            kwargs["artifact_dir"] = Path(v)
        if (v := os.environ.get("EVIDENCE_DIR")) is not None:
            kwargs["evidence_dir"] = Path(v)
        if (v := os.environ.get("LOG_DIR")) is not None:
            kwargs["log_dir"] = Path(v)
        if (v := os.environ.get("LOG_LEVEL")) is not None:
            kwargs["log_level"] = v
        return cls(**kwargs)
