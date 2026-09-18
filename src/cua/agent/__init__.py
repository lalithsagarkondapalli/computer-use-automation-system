"""Discovery agent: the LLM-driven observe -> decide -> validate -> act ->
observe loop (Section 3.1). The loop implementation (`loop.py`) lands in
Phase 4; this package currently holds its domain models (`Observation`,
`AgentAction`), which are also what `cua.models.ModelAdapter` and the
Phase 3 policy engine are typed against.
"""
