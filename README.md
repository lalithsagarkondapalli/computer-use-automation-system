# Computer-Use Automation System

Built solo, end to end: problem framing, architecture, implementation, 229 tests, and a genuine run against a real Anthropic model.

A system that turns "an LLM figuring out how to do something in a legacy
web UI, once" into "a typed, reviewable, deterministic capability that
replays forever without an LLM." In one sentence:

**MODEL DISCOVERS → ARTIFACT CAPTURES → DETERMINISTIC ENGINE REPLAYS → HUMAN INTERVENES WHEN REQUIRED.**

1. **Discover** — a real Anthropic model, given a natural-language goal and a live browser, decides one action at a time (navigate, click, fill, read, checkpoint) against a target web app. Every proposal is validated and policy-checked before it touches the page.
2. **Capture** — once the goal is reached, the executed trace is deterministically compiled into a typed, versioned `CapabilityArtifact` — the steps, the target signals for each control, a checkpoint that proves success, and which values are runtime parameters vs. fixed. No LLM call happens during this step.
3. **Replay** — later, with no model involved at all, `ReplayEngine` reads that artifact and executes it against a fresh browser session with new input, verifying the same checkpoint and extracting the same typed output. Zero LLM calls, deterministic outcome, structured result (`SUCCESS` / `BUSINESS_OUTCOME` / `RECOVERABLE` / `POLICY_BLOCKED` / `ESCALATION_REQUIRED` / `HARD_FAILURE`).
4. **Human intervenes when required** — if replay or discovery hits something automation shouldn't decide alone (an unrecognized dialog, an ambiguous control, a stuck loop), the same live session pauses, a human takes over on the same page, and control hands back — never a second, disconnected session.

## What I owned, and what I learned

**The problem.** An LLM agent that clicks through a legacy web app is impressive once and a liability every day after: each run costs tokens, takes seconds per step, and can decide differently. Operations teams need the same task to run the same way every time, and to stop and ask a human when something unexpected appears.

**Decisions I made**

- **Split "figuring it out" from "doing it".** The model is used once, to discover a workflow; the result is compiled into a typed, human-reviewable YAML artifact; replay never calls a model. An import-graph test enforces that replay *cannot* depend on the model layer, rather than relying on convention.
- **Typed outcomes instead of exceptions.** Every run returns one of six statuses (`SUCCESS`, `BUSINESS_OUTCOME`, `ESCALATION_REQUIRED`, ...), so a caller can route "customer not found" differently from "the page changed".
- **Safety before the page is touched.** Domain/action allowlists and a risk gate run on every proposed action; risk is classified deterministically, never by asking the model how risky its own action is. Secrets and PII are redacted before logging.
- **Humans take over the *same* session.** On an unrecognised dialog or a stuck loop, automation pauses and a person acts on the same live page; resume is validated, not trusted.

**What running it for real taught me** (all preserved in `evidence/`, not edited away):

1. The model's first navigation was a relative URL with no page loaded; the policy engine correctly blocked it. Fix: a trusted base URL for exactly that case, without loosening the allowlist.
2. The model reached the goal but never proposed a checkpoint, so the artifact builder refused to build. Fix: synthesise the checkpoint deterministically from what the run actually read.
3. Credential failures surfaced as clear typed errors rather than retries, which is the behaviour I wanted to prove.

The pattern I'd take into any AI product: let the model explore, make what it found reviewable, and run the hot path without it.

This README gets a reviewer running the system. `REPORT.md` covers the design decisions and trade-offs in depth.

## Architecture overview

```
        genuine, once                          deterministic, every time after
  ┌─────────────────────┐                 ┌──────────────────────────────────┐
  │   DiscoveryLoop      │                 │          ReplayEngine            │
  │  (uses an LLM)        │   produces     │       (never uses an LLM)         │
  │                       │  ───────────▶  │                                   │
  │ observe → decide      │  CapabilityArtifact │ load → validate → interpolate │
  │ → validate → policy   │   (typed, YAML,      │ → policy → act → checkpoint  │
  │ → act → observe       │    human-reviewable) │ → extract → typed RunResult  │
  └─────────┬─────────────┘                 └───────────────┬──────────────────┘
            │                                                │
            └──────────────── same SurfaceAdapter interface ─┘
                          (PlaywrightWebSurface today;
                           the interface is surface-agnostic)

  Both paths share: PolicyEngine (domain/action/risk allowlist),
  HandoffController (same-session human takeover), StructuredLogger
  (redacted JSONL), the RunStatus/OutcomeCode result taxonomy.
```

Package layout (`src/cua/`):

- `actions.py` — the closed action vocabulary (`navigate`, `click`, `fill`, `select`, `read_extract`, `wait`, `checkpoint`, plus the discovery-only `finish`/`request_human`).
- `agent/` — `DiscoveryLoop` (the OBSERVE→DECIDE→VALIDATE→POLICY→ACT cycle), `ArtifactBuilder` (trace → artifact), risk classification, stuck-state fingerprinting, checkpoint synthesis.
- `models/` — `ModelAdapter` interface and the one concrete `AnthropicModelAdapter`. The only place an LLM SDK is imported.
- `artifact/` — the `CapabilityArtifact` schema, `TargetDescriptor` (multi-signal control targeting), checkpoint/extraction specs, YAML load/save.
- `surfaces/` — `SurfaceAdapter` interface and the one concrete `PlaywrightWebSurface`.
- `replay/` — `ReplayEngine`, deterministic; structurally never imports `cua.agent`/`cua.models` (`tests/test_replay_model_isolation.py` proves it by walking the actual import graph, not just by convention).
- `safety/` — `PolicyEngine` (domain/route/action/risk allowlist) and secret redaction.
- `handoff/` — the ownership state machine, `HandoffController`, and a minimal local `OperatorConsole`.
- `observability/` — structured, redacted JSONL logging.
- `results/` — the typed `RunResult` contract and `RunStatus`/`OutcomeCode` taxonomy.
- `demo_app/` — the local synthetic "legacy credit union back office" target this whole system operates on.

## Prerequisites

- Python 3.12+
- A Unix-like shell (commands below are for macOS/Linux)
- An Anthropic API key — **only** if you want to run genuine discovery yourself (Phase 8). Not required for the test suite, for the hand-authored artifact demo, or to inspect/replay the artifact already captured under `evidence/`.

## Installation / setup

```bash
git clone <this repo>            # or unpack the delivered archive
cd computer-use-automation-system
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Browser setup

The system drives a real Chromium via Playwright.

```bash
python -m playwright install chromium
```

This downloads a normal, unpinned Chromium build; nothing in the test suite or `scripts/` requires a special browser path on a typical machine. (`PlaywrightWebSurface.launch()` accepts an optional `executable_path` purely so this same code also runs in sandboxes with a pre-installed, fixed-path Chromium — set `PLAYWRIGHT_EXECUTABLE_PATH` only if `playwright install` isn't an option in your environment.)

### Configuration / environment variables

Copy `.env.example` to `.env` for local reference (`.env` is gitignored and never read implicitly by library code — see `src/cua/config.py`). The variables that matter:

| Variable | Required for | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | genuine discovery only (`scripts/run_evidence.py`) | none |
| `MODEL_NAME` | genuine discovery only | none — must be a current Claude model id |
| `DEMO_APP_BASE_URL` | manual runs against the demo app | `http://127.0.0.1:8000` |
| `ALLOWED_DOMAINS` | overriding the policy allowlist | `127.0.0.1,localhost` |
| `HEADED` | showing the browser window (needed to act as the human in a handoff demo) | `false` |

**API secrets are never committed.** `ANTHROPIC_API_KEY` is read only from the process environment (`os.environ`), never accepted as a CLI argument, never written to `.env`, a log file, an artifact, or the demo app's own subprocess environment, and never printed or interpolated into any string the code produces (see `scripts/run_evidence.py`'s module docstring and `src/cua/safety/redaction.py`, which is applied to every structured log line as a second, independent layer). `.env` and `logs/` are gitignored; nothing under `evidence/` in this repository contains a real key (see the secret-scan note below).

## Running tests

```bash
pytest
```

No API key is required. Every test is either model-free (`FakeModelAdapter`/`FakeSurfaceAdapter`) or, for the real-Chromium tests, drives the local demo app with a scripted fake model — genuine LLM calls only ever happen via `scripts/run_evidence.py`, run once by hand, never from the test suite. 229 tests pass as of this submission.

Isolation tests worth running on their own (they prove, by walking the real import graph, that replay can never accidentally depend on the model layer):

```bash
pytest tests/test_replay_model_isolation.py tests/test_discovery_replay_isolation.py tests/test_handoff_isolation.py -v
```

## Demo workflow

### 1. Hand-authored artifact + deterministic replay (no API key needed)

```bash
python scripts/build_demo_artifact.py   # writes artifacts/member_lookup_balance@1.0.0.yaml
uvicorn cua.demo_app.main:app --host 127.0.0.1 --port 8000   # in a second terminal
```

Then exercise it the same way `tests/test_full_contract_e2e.py` and `tests/test_demo_app_e2e.py` do — load the artifact and hand it to `ReplayEngine` against a live `PlaywrightWebSurface`:

```python
import asyncio
from pathlib import Path
from cua.artifact.store import load_artifact
from cua.observability.logger import StructuredLogger
from cua.replay.engine import ReplayEngine
from cua.replay.evidence import EvidenceCapture
from cua.safety.engine import PolicyEngine
from cua.safety.models import PolicyConfig
from cua.surfaces.playwright_web import PlaywrightWebSurface
from cua.actions import ActionType

async def main():
    artifact = load_artifact(Path("artifacts/member_lookup_balance@1.0.0.yaml"))
    policy = PolicyEngine(PolicyConfig(
        allowed_domains=("127.0.0.1",),
        allowed_actions=tuple(ActionType),
    ))
    logger = StructuredLogger(run_id="manual-replay", log_dir=Path("logs"))
    surface = await PlaywrightWebSurface.launch(headed=False)
    try:
        engine = ReplayEngine(policy, logger, EvidenceCapture(Path("evidence") / "manual"))
        result = await engine.run(artifact, {"member_id": "12345"}, surface)
        print(result.status, result.outputs)   # SUCCESS {'savings_balance': 1234.56}
    finally:
        await surface.close()

asyncio.run(main())
```

Try `member_id="00000"` (not found → `BUSINESS_OUTCOME`/`CUSTOMER_NOT_FOUND`) or `"55555"` (compliance hold → `ESCALATION_REQUIRED`, then a human clearing the same live page and handing control back) to see the other two branches — these are exactly what `tests/test_demo_app_e2e.py` exercises against real Chromium.

### 2. Genuine discovery (real Anthropic model, real browser, real evidence)

```bash
export ANTHROPIC_API_KEY='...'   # your own key, in your own shell — never committed
export MODEL_NAME='...'          # a current Anthropic model id, see
                                  # https://docs.claude.com/en/docs/about-claude/models
python scripts/run_evidence.py
```

This single command performs the entire genuine workflow: starts the demo app itself (no second terminal needed), drives a real `AnthropicModelAdapter` through `DiscoveryLoop` against real Chromium with the model deciding every action, builds and saves a `CapabilityArtifact` from the resulting trace, then — in a **fresh** browser session, with a **different** input than discovery used — deterministically replays that exact artifact and proves the model's call count did not increase during replay. See "Evidence directory" below for what this run already produced.

There is no separate CLI for "just replay an already-saved artifact" (`src/cua/cli/` is a documented stub — see Limitations); the Python snippet above, or `scripts/run_evidence.py`'s own replay phase, is the real, runnable path today.

## Evidence directory

`evidence/` holds real, executed evidence — nothing in it is hand-written to look like a run that happened. It was captured once, genuinely, on 2026-09-18 (all timestamps below are from the run itself, in `evidence/*/*_summary.json` and `evidence/discovery/discovery_trace.jsonl`):

- **`evidence/discovered_member_lookup_balance@1.0.0.yaml`** — the `CapabilityArtifact` a real `claude-sonnet-5`-backed discovery run produced, unedited.
- **`evidence/discovery/`** — `discovery_summary.json` (provider, model, goal, stop reason, decision-call count), `discovery_trace.jsonl` (the full structured log), and screenshots. The genuine successful run: goal *"Look up member 12345 and return their current savings balance"*, provider `anthropic`, model `claude-sonnet-5`, `stop_reason=GOAL_ACHIEVED`, **6 genuine model decision calls**, run id `b84627fd-...`.
- **`evidence/discovery_failure/discovery_failure_summary.json`** — a genuine failed discovery attempt, preserved as-is rather than deleted or hidden (see "Genuine evidence, including failures" below).
- **`evidence/replay_success/`** — `replay_summary.json`, `replay_trace.jsonl`, and a screenshot from replaying the discovered artifact, in a fresh browser session, with `member_id=67890`: `RunStatus.SUCCESS`, `savings_balance: 542.1`, model call count identical before and after replay (6 → 6).
- **`evidence/replay_business_outcome/`, `evidence/replay_escalation/`** — currently placeholders only (`.gitkeep`). The `BUSINESS_OUTCOME`/`CUSTOMER_NOT_FOUND` and `ESCALATION_REQUIRED`/handoff paths are real, implemented, and covered by real-Chromium tests (`tests/test_demo_app_e2e.py`), but were not captured as part of this genuine-Anthropic evidence bundle. This is a known, stated gap, not an oversight — see Limitations.

**Three different kinds of evidence exist in this repository, and they are not interchangeable:**

1. **Genuine Anthropic discovery evidence** (`evidence/discovery*/`, `evidence/discovered_*.yaml`) — a real model made real decisions against a real browser. This is the only place that claim is made.
2. **Deterministic replay evidence** (`evidence/replay_success/`) — real Chromium, zero LLM calls, produced by feeding the genuinely-discovered artifact through `ReplayEngine`.
3. **Automated/mock tests** (everything under `tests/`) — `FakeModelAdapter`/`FakeSurfaceAdapter` or a scripted fake model against a real browser. These prove orchestration and logic correctness and are explicitly documented, in the test files themselves, as **not** genuine LLM evidence (e.g. `tests/test_run_evidence.py`'s module docstring; `tests/test_full_contract_e2e.py` names `FakeModelAdapter` in its own docstring specifically so it's never mistaken for Phase 8 evidence).

### Genuine evidence, including failures

The discovery log (`evidence/discovery/discovery_trace.jsonl`) accumulates every run made against it, not only the final success, so the real path to working evidence is visible, not edited away:

1. A first genuine run hit a real bug: the model's first `NAVIGATE` was a relative route with no page yet loaded to resolve it against, and `PolicyEngine` correctly rejected the resulting empty host (`host '' is not in the allowed domain list`) — `DiscoveryLoop` stopped `POLICY_BLOCKED`. Fixed by giving `DiscoveryLoop` an explicit, trusted target base URL to fall back on for exactly that one case, without weakening the allowlist check itself.
2. Two runs failed on `INVALID_OUTPUT_EXHAUSTED` with a 401 from the Anthropic API (`evidence/discovery_failure/discovery_failure_summary.json` preserves the most recent one) — a credential problem, not a bug in this codebase, included here because it's real and it's what "fail fast and clearly on bad credentials, never guess" looks like in practice.
3. A run reached `GOAL_ACHIEVED` with 6 genuine model decisions but the model never proposed an explicit `CHECKPOINT` action, and `ArtifactBuilder` — correctly — refused to build an artifact with no checkpoint at all. Fixed by having `ArtifactBuilder` deterministically synthesize a checkpoint from the trace's own grounded evidence (the target a successful `READ_EXTRACT` actually read) when the model didn't propose one itself, without weakening the "must have a checkpoint" rule.
4. The final run succeeded end to end and is what `evidence/discovered_member_lookup_balance@1.0.0.yaml` and `evidence/replay_success/` were built from.

Only the single most recent discovery-level failure is retained as a discrete `discovery_failure_summary.json` (that file is overwritten per run) — earlier failures survive only in the cumulative `discovery_trace.jsonl` and, where captured, a screenshot. This is a real limitation of the evidence-capture design, not something worked around for this submission — see Limitations.

## Human handoff

When automation hits something it should not decide alone — an unrecognized dialog blocking a checkpoint, an ambiguous control match, or the same (observation, proposed action) pair repeating several times in a row (stuck-state detection) — `HandoffController` pauses automation (`AUTOMATION_CONTROL → PAUSED_FOR_HUMAN → HUMAN_CONTROL`, two separately-logged transitions), and a minimal local `OperatorConsole` (stdlib `http.server`, no auth, no queue — see Limitations) shows why it stopped, with a screenshot. The human does not act through that console's page; they act directly on the same already-visible browser window automation was using, then click Resume. `HandoffController` validates the resume (for replay, by literally re-checking the condition that triggered escalation; for discovery, by letting the model re-observe and re-decide) before returning control — clicking Resume is never blindly trusted. `tests/test_handoff_playwright_integration.py` proves this against a real browser: escalate → human clears the compliance-hold modal → resume → automation continues and reaches `SUCCESS`, all in the one session.

## Safety / error model overview

Two enforcement points, both before anything touches the live surface: `PolicyEngine.check_domain()` (scheme + host allowlist, checked before *and* after every navigation/click, so a redirect that escapes the allowlist is still caught) and `PolicyEngine.check_action()` (action-type allowlist, plus a `RiskLevel` gate — `SAFE`/`SENSITIVE`/`IRREVERSIBLE`, classified by a small deterministic keyword lookup over the proposal's own target/params, never by asking the model to self-report how risky its own action is). Every run returns one typed `RunResult` with a `RunStatus` from a fixed six-member taxonomy — `SUCCESS`, `BUSINESS_OUTCOME`, `RECOVERABLE`, `POLICY_BLOCKED`, `ESCALATION_REQUIRED`, `HARD_FAILURE` — so a caller branches on structured status, never on catching an exception. Secret- and PII-shaped values (API keys, bearer tokens, card-number-length digit runs, SSNs) are redacted before anything reaches a log line or an `InterventionRequest`. See `REPORT.md` §6 for the full picture.

## Repository structure

```
src/cua/            the system itself (see Architecture overview above)
tests/               pytest suite (model-free + real-Chromium; no API key needed)
scripts/
  build_demo_artifact.py   hand-authors + saves the Phase 3/6 demo artifact
  run_evidence.py           the one genuine, real-model Phase 8 workflow
artifacts/           hand-authored capability artifacts
evidence/            real, executed discovery/replay evidence (see above)
README.md, REPORT.md this file and the design write-up
```

## Limitations / intentional cuts

- **No production CLI.** `src/cua/cli/` is a documented stub; there is no `cua discover`/`cua replay` command. Real entry points today are `scripts/run_evidence.py`, `scripts/build_demo_artifact.py`, and direct library use (see the demo workflow above and the test suite).
- **One target surface implemented.** `SurfaceAdapter` is designed to be surface-agnostic (see `REPORT.md` §4), but only `PlaywrightWebSurface` (web) exists. No desktop/accessibility-tree surface is built.
- **One tenant.** `TargetIdentity.compatibility_fingerprint` and `CapabilityArtifact.variant_overrides` exist in the schema for future multi-tenant drift handling but are unused — see `REPORT.md` §4.
- **`OperatorConsole` is intentionally minimal.** No auth, no TLS, no multi-operator queue, no persistent dashboard — it renders one page and handles one POST. Fine for a same-machine human-in-the-loop demo, not for production.
- **Genuine evidence covers the success and target-fix/checkpoint-fix failure paths, not every branch.** `BUSINESS_OUTCOME` and `ESCALATION_REQUIRED` are real, implemented, and covered by real-browser tests, but were not captured with a genuine Anthropic-backed discovery run as part of this evidence bundle.
- **Only the most recent discovery-level failure is retained as a discrete summary file** (`evidence/discovery_failure/`); it is overwritten per run. Full history lives only in the cumulative structured log.
- **Logging is per-run JSONL files on local disk**, not a real observability backend (no aggregation, no dashboards, no alerting).
