"""Deterministic replay engine.

Given a `CapabilityArtifact` and typed runtime parameters, executes the
recorded steps against a `SurfaceAdapter` with zero model calls. This is
enforced, not just documented -- see tests/test_replay_model_isolation.py,
which fails if anything reachable from this package ever imports
`cua.agent`, `cua.models`, `anthropic`, or `openai`.
"""
