# 2026-10-08 DELTA maintenance — blocked/partial

I'm an AI agent (Codex) helping Engel Nyst (@enyst) with project work.

Two new enterprise surfaces now have explicit **Not mapped** reasons and exact live prerequisites. This is source-verified coverage accounting, with **zero live passes**. The accepted baseline remains unchanged. This run does not include the daily smoke or rotation tiers.

- Repository: `OpenHands/OpenHands`; origin verified as `enyst/playground`, not canonical.
- BASE: `ed815e141409c7b52991ab8d13c553b595da9165` (2026-10-06); TARGET: `baf1cbef090fce793d3017276c1025c057920a4b`.
- Complete history, ancestor check valid; 14 commits / 77 changed paths. All 14 resolve through the owning repository's commits/pulls API to merged PRs: 14 documented intents, 0 undocumented, 0 contradictory. Private OHE acceptance criteria for #17988/#17969 remain unavailable; their descriptions document intent.
- Source scope expands to F01–F27 because #18051's HTTP-error parser feeds the common QueryCache/MutationCache. This is conservative delta widening; it does not claim every happy path changed. All 77 paths, all 8 shared paths and all 4 initially unmapped paths are accounted for in `source-scope.json`; Playwright configuration is non-user-facing harness work.

## Coverage and prerequisites

| Evidence type | Pass | Fail | Blocked | Not-run | Denominator |
|---|---:|---:|---:|---:|---|
| Live mapped behavior inventory | 0 | 0 | 736 | 0 | 736 mapped sub-features in 27 families |
| Live families completed | 0 | 0 | 27 | 0 | 27 affected families |
| Static map check / coverage / testids / baseline / affected | 5 | 0 | 0 | 0 | 5 commands |

The behavior inventory is a list of required obligations, not an executed entry-point/viewport matrix. **No live entry point, viewport, doctor or backend scenario was exercised.** The complete private inventory records each stable ID, expected behavior, literal family recipe, prerequisites, planned entry points and the launch blocker. No run was allocated, so CLI `evidence add/report` has no valid run; no artificial run metadata or screenshots were created. Viewport and actual selected backend/scenario must be explicit in every evidence entry when live work resumes.

Coordinator normal launch stopped before allocation: `Only 874 MB of memory available; a run needs about 1500 MB.` A reduced-threshold attempt also stopped at 79 MB. Automatic approval review rejected disabling the guard (`--min-free-mb 0`) because a resource-heavy stack could cause severe system-wide memory pressure; the rejected action was not executed or bypassed. An independent fresh default-threshold launch at 07:03 UTC stopped at 117 MB. Live checks require safe memory headroom that passes the normal guard, or explicit approval for the rejected override.

No scheduled `DEEPSEEK_API_KEY` or active project environment/key source was found. Real-model turns, event/tool outcomes and recipes requiring configured DeepSeek credentials are additionally blocked by that key prerequisite. No provider substitution or mocked live proof was used. DeepSeek spend: **USD 0 of USD 10**.

## Changed rows and retained limits

| Index correction | Expected / actual proof | Live result |
|---|---|---|
| Enterprise Super Admin setup guide (#17969) | Exact Cloud `/me` permission and genuine setup-state gates agree with source; the new directory is now explicitly excluded from coverage. Positive guide progress, links, collapse/reopen, server removal and desktop/phone states were not driven. | blocked |
| Suspended Cloud workspace recovery (#17988) | Exact selected-org 403 detail strings and another accessible workspace are documented. Genuine organization/membership suspension, switching and request isolation were not driven. | blocked |

Neither exclusion retires a feature or treats an account gap as a pass. Existing IDs/counts remain 27 families / 736 sub-features; all 999 cited test IDs resolve. Coverage now reports setup-guide as an explicit prerequisite exclusion; this **does not mean it has a proven recipe**.

Issue audit checked 65 owning-repository references: 46 issues (40 open, 6 closed) and 19 PR references. #18094 closed after BASE and requires F05.plus-menu, F07.status-menu and F08.tabs-menu retests. Automation #551 closed before BASE but still has failure wording: F24.encryption needs set/rotate/clear + sync without dirtying an existing exported automation, since the old recipe can mask the fixed behavior. Their known-failure text remains unchanged until live verification. Further source candidates retained without runtime claims: F25's 10s health wording versus Local 30s / Cloud 5m, the DNT/PostHog failure note after #18086, and F27 calling closed #17562 open. Details and exact entry points/prerequisites are in `issue-closures.json` and `intent.md`.

No new defect was independently reproduced, so no new issue or causal regression claim was filed. No product code, schedules or user state changed.

## Environment, review and teardown

macOS; Node 25.9.0, npm 11.12.1, uv/uvx 0.11.19, installed Google Chrome 154.0.8037.98. Locked dependencies installed with `npm ci --ignore-scripts`; npm emitted Node 25 engine warnings for jsdom/vitest. No build or runtime result is inferred from installation. TARGET pins Agent Server 1.53.0 and automation 1.19.0; no backend was started and no ports were allocated. The active configured identity is `smolpaws`, upstream push=false, verified `smolpaws/OpenHands` fork push=true. Commit author is user-authorized enyst; actual committer/signing configuration is preserved.

Independent source review accepted both exclusions and independently reran the static checks. This two-bullet documentation change needs no design diagram. #17568 is closed but retains `ready-for-dev`; its acceptance criteria explicitly own maintained feature maps and honest account gaps, and the current PR-description gate checks the label without requiring an open state. It is referenced as context, not claimed fixed again. No `codex` or `codex-automation` label exists upstream.

All attempted launches stopped before run allocation; coordinator and independent `runs` inventories are empty. There are no owned stack processes, ports, browser sessions, fixtures or conversations to stop. Reviewed artifacts survive under this directory. Raw GitHub source, complete inventory and private notes remain outside GitHub. CI/review status belongs to the PR; this draft must not be merged as a completed delta. **BASE is retained, and no next baseline is proposed.**
