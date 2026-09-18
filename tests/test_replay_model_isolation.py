"""The automated proof that deterministic replay makes zero LLM decisions.

Two independent checks, deliberately redundant with each other:

1. A static, transitive AST walk over every module reachable from
   `cua.replay`'s own source files, asserting none of them ever import
   `cua.agent`, `cua.models`, `anthropic`, or `openai`. This catches a
   violation anywhere in the dependency graph, not just a direct import in
   a replay module itself.
2. A subprocess that imports `cua.replay.engine` in a brand-new
   interpreter and inspects `sys.modules` afterward. This is immune to
   whatever the rest of the test suite happened to import earlier in the
   same process (which would otherwise make a pure in-process
   `sys.modules` check meaningless), so it is a true "importing replay
   alone pulls in nothing agent/model-related" check.

If either check fails, replay is no longer structurally isolated from the
model -- fix the dependency, don't loosen this test.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FORBIDDEN_MODULES = ("cua.agent", "cua.models", "anthropic", "openai")


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _module_path(dotted: str) -> Path | None:
    parts = dotted.split(".")
    as_module = SRC_ROOT.joinpath(*parts).with_suffix(".py")
    if as_module.exists():
        return as_module
    as_package = SRC_ROOT.joinpath(*parts, "__init__.py")
    if as_package.exists():
        return as_package
    return None


def _forbidden_hit(dotted: str) -> str | None:
    for forbidden in FORBIDDEN_MODULES:
        if dotted == forbidden or dotted.startswith(forbidden + "."):
            return forbidden
    return None


def test_replay_source_never_transitively_imports_model_or_agent_code():
    replay_dir = SRC_ROOT / "cua" / "replay"
    frontier = list(replay_dir.rglob("*.py"))
    visited: set[Path] = set()
    violations: list[str] = []

    while frontier:
        path = frontier.pop()
        if path in visited:
            continue
        visited.add(path)
        for imported in _imports_of(path):
            forbidden = _forbidden_hit(imported)
            if forbidden is not None:
                violations.append(f"{path.relative_to(SRC_ROOT)} imports {imported!r} (forbidden: {forbidden!r})")
                continue
            if imported.startswith("cua."):
                dependency_path = _module_path(imported)
                if dependency_path is not None and dependency_path not in visited:
                    frontier.append(dependency_path)

    assert not violations, "replay must never (transitively) import model/agent code:\n" + "\n".join(violations)


def test_importing_replay_engine_in_a_fresh_interpreter_loads_no_model_or_agent_modules():
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
