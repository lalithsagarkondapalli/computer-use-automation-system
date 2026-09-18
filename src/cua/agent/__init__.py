"""Discovery agent: the LLM-driven observe -> decide -> validate -> act ->
observe loop (Section 3.1). The loop implementation (`loop.py`) lands in
Phase 4; this package currently holds its decision-side domain model,
`AgentAction`. `Observation` (what a surface reports) lives in
`cua.surfaces.observation`, not here -- see that module's docstring for
why: it keeps `cua.surfaces` (and therefore `cua.replay`) free of any
dependency on agent/model code.
"""
