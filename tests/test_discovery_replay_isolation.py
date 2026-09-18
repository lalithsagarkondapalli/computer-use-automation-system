"""Phase 4's own architectural isolation test (Section 8 of the Phase 4
brief): discovery may depend on the model layer; replay must remain
unable to, even now that `cua.agent`/`cua.models` contain real provider
code (`anthropic`) instead of Phase 2/3's empty interfaces.

`tests/test_replay_model_isolation.py` (unmodified since Phase 3) already
re-runs its own static-AST-walk and fresh-subprocess checks against
today's larger source tree every time the suite runs -- its
`FORBIDDEN_MODULES` tuple already includes `anthropic`, so it is not a
vacuous check now that `anthropic` is a real, installed dependency. This
file adds the two things that test does not cover: (a) a positive check
that discovery *does* depend on the model layer, so a reader can trust
that the isolation test above is guarding something real rather than
passing by accident because nothing imports `anthropic` at all; and (b)
the reverse-direction layering check -- `cua.agent`/`cua.models` never
import `cua.replay` either, so there is no accidental cycle.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _all_imports_under(package_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in package_dir.rglob("*.py"):
        result |= _imports_of(path)
    return result


def test_discovery_does_transitively_depend_on_the_model_layer():
    """Sanity check: if this ever went false, test_replay_model_isolation's
    "replay never imports cua.agent/cua.models/anthropic/openai" guarantee
    would be trivially true for the wrong reason (nothing imports them at
    all), not because the boundary is actually being enforced."""
    agent_imports = _all_imports_under(SRC_ROOT / "cua" / "agent")
    assert any(name == "cua.models.base" or name.startswith("cua.models") for name in agent_imports)
    assert any(name.startswith("anthropic") for name in _all_imports_under(SRC_ROOT / "cua" / "models"))


def test_agent_and_model_packages_never_import_replay():
    """Layering should hold in both directions: discovery depends on
    models/surfaces/artifact/safety, never on cua.replay. (The
    discovery -> artifact -> replay pipeline is real, but it is composed
    by a *caller* -- a CLI/test/demo script -- not by an import inside
    cua.agent or cua.models themselves; see
    tests/test_discovery_to_replay_integration.py.)"""
    violations = []
    for package in ("agent", "models"):
        for imported in _all_imports_under(SRC_ROOT / "cua" / package):
            if imported == "cua.replay" or imported.startswith("cua.replay."):
                violations.append(f"cua.{package} imports {imported!r}")
    assert not violations, "\n".join(violations)


def test_replay_isolation_holds_with_the_full_phase4_source_tree_present():
    """Re-runs the Phase 3 subprocess check verbatim, now that
    src/cua/agent/loop.py, src/cua/agent/artifact_builder.py, and
    src/cua/models/anthropic_adapter.py all exist and are on disk --
    the risk Phase 4 specifically introduces is that importing
    cua.replay.engine could, through some new shared module, drag in a
    real provider SDK. It does not."""
    code = (
        "import sys\n"
        "sys.path.insert(0, 'src')\n"
        "import cua.replay.engine\n"
        "forbidden = sorted(m for m in sys.modules if m.startswith(("
        "'cua.agent', 'cua.models', 'anthropic', 'openai'))) \n"
        "print(forbidden)\n"
        "assert not forbidden, forbidden\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
