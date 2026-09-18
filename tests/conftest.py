"""Shared fixtures for real end-to-end tests against the demo app.

Factored out in Phase 7 so both tests/test_demo_app_e2e.py (Phase 6) and
tests/test_full_contract_e2e.py (Phase 7) launch the real app the same
way, instead of two independent copies of the same subprocess-management
code drifting apart.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def demo_app_base_url():
    """Starts the real demo app as a real subprocess (not in-process, not
    mocked) on an ephemeral port, and tears it down after this module's
    tests finish."""
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cua.demo_app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(REPO_ROOT / "src"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/search", timeout=1)
            break
        except Exception as exc:  # noqa: BLE001 -- readiness poll, retry until deadline
            last_exc = exc
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError(f"demo app never became ready: {last_exc}")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
