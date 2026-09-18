# Computer-Use Automation System

Take-home submission for interface.ai — Applied AI Engineer / Business Automation.

**Status: under active, phased development.** This file will be replaced with
the full setup/usage/demo guide once the vertical slice is complete
(discovery → artifact → deterministic replay → escalation, all demonstrated
with real evidence). Until then, treat this as an orientation stub, not a
finished deliverable.

## Repository layout (current)

- `src/cua/` — the automation system itself:
  - `actions.py` — the shared, closed action vocabulary (`navigate`, `click`, `fill`, `select`, `read_extract`, `wait`, `checkpoint`, plus the agent-loop-only `finish` / `request_human`).
  - `artifact/` — the `CapabilityArtifact` schema, `TargetDescriptor`, checkpoint/extraction specs, YAML store.
  - `agent/` — discovery-loop domain models (`Observation`, `AgentAction`). The loop itself lands in Phase 4.
  - `models/` — `ModelAdapter` interface (the only door an LLM can walk through). Concrete Anthropic adapter lands in Phase 4.
  - `surfaces/` — `SurfaceAdapter` interface. The Playwright implementation lands in Phase 3.
  - `safety/` — policy config/decision models + redaction. The enforcing `PolicyEngine` lands in Phase 3.
  - `observability/` — structured JSONL log event schema + logger.
  - `results/` — the typed `RunResult` contract and the six-category error taxonomy.
  - `replay/`, `handoff/`, `cli/`, `demo_app/` — placeholders; implemented in Phases 3, 5, and 6.
- `tests/` — pytest suite for everything implemented so far.
- `artifacts/` — saved capability artifacts (empty until a real discovery run produces one).
- `evidence/` — evidence of real discovery and replay runs. Empty until Phase 8; see `evidence/README.md`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

No API key or browser is required to run the current test suite — Phase 2
covers configuration, the artifact schema, the result/error taxonomy, and
the (currently model-free, browser-free) domain models and interfaces.
