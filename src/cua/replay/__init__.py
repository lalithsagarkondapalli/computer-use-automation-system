"""Deterministic replay engine.

Given a `CapabilityArtifact` and typed runtime parameters, executes the
recorded steps against a `SurfaceAdapter` with zero model calls. Implemented
in Phase 3, immediately after the surface adapter and policy engine it
depends on.
"""
