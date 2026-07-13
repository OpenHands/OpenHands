# Live Evidence for Issue 15022 / PR 15023

Final run: `.pr/issue-15022/runs/20260713T113553Z`

## Environment

- Local enterprise app process, started by `uvicorn live_harness_app:app`.
- Local SQLite database seeded with isolated test users/orgs only.
- Local FastAPI stub for LiteLLM admin/model verification plus agent-server endpoints.
- No production organization, production database, production LiteLLM, or production sandbox was used.
- Raw local key strings and localhost endpoint ports are redacted in saved JSON/log artifacts; key comparisons use 16-character SHA-256 fingerprints.
- The harness copies its support scripts to a temporary directory before checking out `origin/main`, so each app process imports production code from the exact checked-out SHA while the evidence runner remains available.

## Command

```bash
OPENHANDS_SUPPRESS_BANNER=1 poetry --project enterprise run python \
  .pr/issue-15022/run_live_evidence.py \
  --main-ref origin/main \
  --pr-ref origin/pr-15023 \
  --restore-ref managed-llm-key-refresh-15022
```

The harness checks out `origin/main`, runs the local app/stub/db flow, restores the PR branch, then repeats against `origin/pr-15023`.

## SHAs

- Current main: `3949e1cc17d9443f1f4ef7d34d428baf065cd919`
- PR head: `7e5f5ebc5ab04371f0d23e1b7a43df9d0e7fc8bc`

## Result

Current main fails the issue criterion:

- Explicit managed refresh endpoint probe: HTTP `405`.
- Upstream managed LiteLLM key was deleted while DB still referenced fingerprint `ff3dd4e36e79dffa`.
- Conversation startup reached the agent-server with that stale fingerprint and the stub rejected it with `token_not_found_in_db`.
- Start task ended `ERROR`.

PR head satisfies the criterion:

- Explicit refresh endpoint returned HTTP `200` and persisted a replacement for managed-refresh user fingerprint `2690c6ca4c52787a`.
- Startup self-heal verified old managed-start fingerprint `ff3dd4e36e79dffa`, received `401`, rotated/persisted replacement fingerprint `aa6e3cef4774eb9e`, and sent that replacement to the agent-server.
- Agent-server accepted the request and the start task ended `READY`.
- BYOK refresh returned HTTP `400`; BYOK LLM fingerprint `5b45457397875fe8` and BYOR fingerprint `110a776e32b16e96` remained unchanged.

## Artifacts

- Summary: `.pr/issue-15022/runs/20260713T113553Z/summary.json`
- Main observations: `.pr/issue-15022/runs/20260713T113553Z/main/observations.json`
- PR observations: `.pr/issue-15022/runs/20260713T113553Z/pr/observations.json`
- Focused main logs: `.pr/issue-15022/runs/20260713T113553Z/main/key-events.log`
- Focused PR logs: `.pr/issue-15022/runs/20260713T113553Z/pr/key-events.log`
- Full sanitized app/stub logs are in each `main/` and `pr/` run directory.
