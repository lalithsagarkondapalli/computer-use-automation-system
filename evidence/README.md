# Evidence

This directory holds real, executed evidence — never fabricated — of the
system running end-to-end. It is empty (aside from placeholder files) as of
Phase 2 of the build. It gets populated in Phase 8, after the discovery
agent, replay engine, and handoff mechanism all exist and have actually
been run.

- `discovery/` — logs + captured evidence from a genuine LLM-driven
  discovery run against the local demo app, plus the `CapabilityArtifact`
  it produced.
- `replay_success/` — logs from a deterministic replay of that artifact
  with a valid `member_id`, reaching `SUCCESS` with the extracted balance.
- `replay_business_outcome/` — logs from a replay with an unknown
  `member_id`, reaching `BUSINESS_OUTCOME: CUSTOMER_NOT_FOUND` — a correct
  answer, not a crash.
- `replay_escalation/` — logs from a replay that hits an unexpected
  "compliance hold" modal, is correctly classified `ESCALATION_REQUIRED`,
  and shows a human taking control of the same live session and handing
  it back.

Each subfolder will contain a structured JSONL log and at least one
screenshot once it is populated. If, when the submission is finalized, any
of these folders is still empty or contains only this note, that means the
corresponding run has not actually been executed yet — the README/REPORT
will say so explicitly rather than imply otherwise.
