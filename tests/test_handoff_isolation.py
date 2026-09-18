"""Phase 5's architectural isolation test: `cua.handoff` must be a true
leaf with respect to the model/replay boundary, since Phase 5 wires it
into BOTH `cua.agent` (discovery) and `cua.replay` (deterministic replay).
If `cua.handoff` itself imported `cua.agent`/`cua.models`/`anthropic`/
`openai`, then `cua.replay` (which now imports `cua.handoff.controller`)
would transitively depend on them too -- exactly the violation
tests/test_replay_model_isolation.py exists to catch. This file checks
the specific new edge Phase 5 adds, and test_replay_model_isolation.py
(rerun, unmodified, in this same suite) is the end-to-end proof that the
overall guarantee still holds.

It also checks the "same live session" architectural guarantee (Section
3): `cua.handoff` never imports the one module capable of launching a
browser (`cua.surfaces.playwright_web`), so there is no code path inside
the handoff package that could construct a second session.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FORBIDDEN_FOR_HANDOFF = ("cua.agent", "cua.models", "cua.replay", "cua.surfaces.playwright_web", "anthropic", "openai")


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_handoff_package_imports_nothing_forbidden():
    violations = []
    for path in (SRC_ROOT / "cua" / "handoff").rglob("*.py"):
        for imported in _imports_of(path):
            if any(imported == forbidden or imported.startswith(forbidden + ".") for forbidden in FORBIDDEN_FOR_HANDOFF):
                violations.append(f"{path.relative_to(SRC_ROOT)} imports {imported!r}")
    assert not violations, "\n".join(violations)


def test_replay_isolation_still_holds_with_handoff_wired_in():
    """Re-run of the Phase 3/4 subprocess check, now that cua.replay.engine
    imports cua.handoff.controller -- the exact new edge that could have
    reintroduced a model dependency if cua.handoff were not a clean leaf."""
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


def test_replay_does_transitively_depend_on_handoff():
    """Sanity check the test above isn't vacuous: cua.replay really does
    import cua.handoff now (so 'no forbidden module' above is a real
    guarantee about a real, exercised edge, not an accident of nothing
    being wired up)."""
    imports: set[str] = set()
    for path in (SRC_ROOT / "cua" / "replay").rglob("*.py"):
        imports |= _imports_of(path)
    assert any(name.startswith("cua.handoff") for name in imports)
