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
python -m playwright install chromium
pytest
```

No API key is required to run the test suite — every test either runs
model-free (`FakeModelAdapter`/`FakeSurfaceAdapter`) or, for the real-
Chromium tests, drives the local demo app with a scripted fake model.
`python -m playwright install chromium` downloads a normal, unpinned
Chromium; nothing in the test suite or `scripts/` requires a special
browser path on a normal machine.

## Phase 8: genuine LLM discovery evidence

`scripts/run_evidence.py` performs the one genuine, real-model workflow
this project requires: a real `AnthropicModelAdapter` drives `DiscoveryLoop`
against a real Chromium browser and the local demo app, with the model
(not a script) deciding every action; on success it builds and saves a
`CapabilityArtifact` under `evidence/`, then deterministically replays that
exact artifact in a fresh browser session with a different input and
verifies the typed output -- with zero model calls during replay.

```bash
export ANTHROPIC_API_KEY='...'   # your own key, in your own shell -- never committed
export MODEL_NAME='...'          # a current Anthropic model id, see
                                  # https://docs.claude.com/en/docs/about-claude/models
python scripts/run_evidence.py
```

The script starts and stops the demo app itself (no second terminal
needed), reads the API key only from the environment, and never writes it
anywhere. See the script's own module docstring for the full credential-
handling guarantees. `tests/test_run_evidence.py` exercises this script's
orchestration logic (credential handling, subprocess environment hygiene,
artifact parameterization, the replay zero-model-call boundary) without a
real key -- that is explicitly not genuine Phase 8 evidence, and is never
presented as such.
