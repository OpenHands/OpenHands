I'm an AI agent (Codex) helping Engel Nyst (@enyst) with project work.

# Delta maintenance: partial, baseline retained

BASE: 553d4519113192a80fb18376fdb6c1e39fa06387.
TARGET: 5a8c3832154b14d2dad651801c0ab3f74c96f02e.

This is a DELTA pass, without smoke or weekly rotation. Twelve first-parent commits changed 172 paths. The SDK/client and automation version changes plus shared consumers widen required verification to F01–F27 (745 stable IDs). Every interval commit resolved through the owning commit-to-PR API; descriptions, files, review discussion and linked criteria were read. Intent remains separate from runtime outcomes.

Covered merged PRs: [18228](https://github.com/OpenHands/OpenHands/pull/18228), [18219](https://github.com/OpenHands/OpenHands/pull/18219), [18243](https://github.com/OpenHands/OpenHands/pull/18243), [17758](https://github.com/OpenHands/OpenHands/pull/17758), [18237](https://github.com/OpenHands/OpenHands/pull/18237), [18175](https://github.com/OpenHands/OpenHands/pull/18175), [18253](https://github.com/OpenHands/OpenHands/pull/18253), [17199](https://github.com/OpenHands/OpenHands/pull/17199), [17200](https://github.com/OpenHands/OpenHands/pull/17200), [17201](https://github.com/OpenHands/OpenHands/pull/17201), [17242](https://github.com/OpenHands/OpenHands/pull/17242), [17688](https://github.com/OpenHands/OpenHands/pull/17688).

## Changes and live outcomes

- F22.agent-assisted-setup: replaces the broad draft instructions with a scoped credential-free new-form/save/reload/resume/custom-script recipe. Root opened via Dashboard → Add → Create without a conversation; name/prompt/yearly cron persisted. Desktop dispatched the generated callback-only script; its final successful state was inspected after phone reload. Phone draft resume/rename persisted in the UI and GET. Root ledger: 3 scoped passes, 1 model-budget block. Root recorded these rows after switching to phone; the ledger's then-current viewport/URL is not capture metadata. The original [desktop save video](draft-root/draft-desktop.mp4) is 1440×1000; [phone resume video](draft-root/resume-phone.mp4) and [phone renamed draft](draft-root/resumed-phone.png) are 390×844. Neither draft save nor a Pending run proves LLM execution.
- F24.encryption: removes the stale Known failure for automation#551. A fresh independent family run on pinned automation1.19.3 verified 26 IDs through 28 scoped checks: 25 pass, 0 fail, 2 Cloud blocks, 1 nonmock status-error prerequisite not-run. It includes desktop/phone unsupported UI on a separate automation1.7.1 stack. Root then independently repeated all six key-only set/rotate/clear states on another fresh pinned stack: 2 entry scopes pass. No automation PATCH intervened; every key-only save dirtied the export, and the next sync created a new commit containing ciphertext, changed ciphertext, then plaintext. [Desktop export readbacks](encryption-root/desktop-export-results.json), [phone readbacks](encryption-root/phone-export-results.json), [desktop recording](encryption-root/desktop-key-only.mp4), [phone recording](encryption-root/phone-key-only.mp4). The family entry used the dashboard Git Sync button and direct URL; a sidebar-label in the raw peer ledger does not prove the sidebar click.
- README: records the unverified new home Automate handoff, split setup/form-tool boundaries, dashboard draft management, replacement F21/F23 editor, Pi/OpenCode ACP presets and Cloud org handback. These are source-grounded gaps with prerequisites, not successful live recipes. OpenCode credentials are optional; genuine ACP execution and provider authorization are still separate checks.
- F26: 7 real terminal command scopes across 5 IDs pass (version/-v, info, help, two flag-conflict cases, missing-build scratch copy). These are native CLI proof only, not Docker, Electron or library-host proof.

The full required-ID inventory and complete append-only ledgers remain private. Executed scopes overlap IDs and are not a completed 745-ID viewport/backend matrix. The remaining affected entry points retain blocked or not-run statuses; there is no whole-family pass inferred from one successful example.

## Fresh F22 rerun

A separate fresh stack drove all 29 F22 IDs through 40 scoped checks: 25 pass, 3 fail, 7 blocked, 5 not-run. It independently confirmed desktop and fresh phone draft save/reload, resume/rename and phone inline Delete, with real backend readbacks and zero model calls. Broader credential-free template/import/form/responder checks passed. Existing failures [#17959](https://github.com/OpenHands/OpenHands/issues/17959) (blank no-match search) and [#18061](https://github.com/OpenHands/OpenHands/issues/18061) (blank Default after kind switches) were reproduced; no duplicates were filed. The third failure is the new Test runs stale-Pending candidate, independently confirmed on a second fresh stack before filing. Full phone behavior/Cloud paths and model-dependent creation/execution remain unverified.

The second fresh reproduction adds one failed desktop Local/no-conversation scope: API COMPLETED before and after a 45.158-second exact Successful-row wait; the visible row remained Pending, doctor was healthy, and reload showed Successful. [First backend receipt](../../bugs/2026-10-10/draft-run-status/first/sanitized-run-summary.json) and [second receipt](../../bugs/2026-10-10/draft-run-status/second/second-sanitized-run-summary.json) record actual timestamps. Origin remains unconfirmed; no pre-change live comparison or causal PR is claimed.

| Producer/scoped checks | Pass | Fail | Blocked | Not-run |
|---|---:|---:|---:|---:|
| Root draft scopes | 3 | 0 | 1 | 0 |
| Fresh F22 family peer | 25 | 3 | 7 | 5 |
| Second F22 defect peer | 0 | 1 | 0 | 0 |
| F24 family peer (including old backend) | 25 | 0 | 2 | 1 |
| Root fresh F24 confirmation | 2 | 0 | 0 | 0 |
| No-stack native CLI | 7 | 0 | 0 | 0 |
| Total executed/explicit scopes (82) | 62 | 4 | 10 | 6 |

These scopes touch 60 distinct IDs across F22/F24/F26; that is not 60 complete-ID passes. Source coverage is 27/27 families. Live family coverage is partial for two UI families plus no-stack CLI scopes; no full-delta family pass or baseline completion is claimed.

## Closure checks, prerequisites and limits

All 89 genuine map references were audited in their owning repositories, including SDK and automation. Automation#551 is now verified fixed at the actual Canvas pin1.19.3 ([fix#563](https://github.com/OpenHands/automation/pull/563)). F23.run-logs-modal's closed Canvas#18173/fix#18175 still needs a genuine long-output desktop/phone Logs-dialog drive; its old expectation is retained. Open linked defects are not waived or duplicated.

Model spend is USD0 of USD10, with zero calls, validation requests or real model-key reads. Scheduled DEEPSEEK_API_KEY is absent; an existing user-configured project key source is recorded but unused. Genuine model paths remain blocked until aggregate parent/child/title/planner/judge and automation producer costs can be bounded confidently. No provider substitution or mocked LLM response was used as live proof.

Cloud checks need genuine accounts/permissions: organization handback with multiple accessible organizations, enterprise setup-state/tour entitlements, a member without manage_automations and two organizations for Git Sync conflict. Native/Docker/embedded and other affected recipes not reached by these serial stacks remain unverified. F24.error-state specifically needs a genuine non-404 status failure while automation health remains healthy.

Unmapped-path accounting: scripts/check-sdk-version-sync.mjs is a contributor gate; acp-brand-marks.ts belongs to ACP icon consumers; automation-form.ts is shared by composer/conversation and automation form consumers. F22's Source includes its form constant/session/service; the other new distinct behaviors are explicitly not live-mapped.

## Validation and teardown

Locked npm dependencies installed in an automation-owned managed checkout. Node25.9.0/npm11.12.1, uv0.11.19, macOS27.0.1/Chrome155.0.8059.39; production Canvas1.26.0 built at TARGET. Production SDK1.54.0/automation1.19.3, OpenHands-Neutral; font metrics were not independently enumerated. Dependency engine warnings on Node25 are recorded; the real build and doctor passed.

Map check745 IDs/27 families, coverage31 routes/29 feature directories (documented context-menu exclusion), test IDs all resolved, baseline ancestry and explicit BASE/TARGET affected commands pass. These are static checks, not UI proof.

All finished owned stacks were stopped with the CLI, their recorded PIDs were absent and ports were checked under host execution (sandbox EPERM is not connection-refused proof). Drafts, automations and materialized script tests were removed, Git Sync was disabled/cleared, and original media survived stop. One peer browser-tab closure recovered on a healthy doctored stack; origin is unconfirmed and no Canvas defect is inferred. Primary checkout and other agents' resources were preserved. Automatic approval review rejected the first F22 peer's UI deletion of OPENHANDS_URL because evidence that it was created by this run was insufficient. That app-origin fixture remains solely in the stopped private run home; no live service or user-shared resource remains. It was not deleted through a workaround. Private keys, homes, raw logs, snapshots and ledgers are excluded from this PR.

No baseline advancement is proposed. The accepted BASE remains unchanged because the full delta, distinct entry points and F23 closure retest are unfinished.
