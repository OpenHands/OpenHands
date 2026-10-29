I am an AI agent helping Engel Nyst (@enyst) with project work.

# Weekly verification maintenance — reviewed report

**Verdict: changed and partial.** Source review and scoped live observations span 28/28 feature families. The frozen map has 779 stable behavior IDs. All live workers have finished. Capacity-blocked packaging and unavailable branches are not invented as passes; coverage attempts and cleanup are complete; the draft fork maintenance PR is published. Snapshot: 2026-10-17T22:18:33.083173+00:00.

Accepted BASE `d3a58cd1cc045bbb1b24f2c8c0d79a8404ef239b`; frozen TARGET `61a3bd93e642a0ff4def73b29496a73245628209`. SDK 1.55.1, automation 1.20.1. No accepted-baseline advancement.

**733/779 IDs have a recorded observation:** 633 pass-only observations, 88 existing-known product failures, 4 newly filed product failures, 0 IDs classified unresolved, and 8 fix assertions. Separately: 46 blocked-only, 0 not-run-only and 0 unreconciled IDs. These are mutually exclusive reviewed categories, not complete passes for every branch of an observed ID.

Failure taxonomy reviews both actual `fail` and `xfail`: newly filed > existing known > invalid-measurement-only; unresolved stays explicit. Two invalid initial hook measurements remain failed checks; corrected measurements reproduce the known hook issue, so that ID is existing-known. The raw highest-result label `xfail` alone must not be mistaken for the count of all known product failures.

**Latest 1554 recorded checks, original result labels kept separate:** pass 1157, fail 146, xfail 14, xpass 13, blocked 98, not-run 126. These count entry-point rows, not stable IDs. No fail is relabeled pass. [Classification detail](failure-summary.json).

| Family | IDs | Pass observed | Existing failure | Filed failure | Unresolved | Fix assertion | Blocked only | Not-run only | Unreconciled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | 22 | 18 | 1 | 0 | 0 | 0 | 3 | 0 | 0 |
| F02 | 28 | 23 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| F03 | 42 | 35 | 4 | 0 | 0 | 0 | 3 | 0 | 0 |
| F04 | 41 | 35 | 4 | 0 | 0 | 0 | 2 | 0 | 0 |
| F05 | 25 | 20 | 3 | 0 | 0 | 2 | 0 | 0 | 0 |
| F06 | 36 | 31 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| F07 | 25 | 20 | 4 | 0 | 0 | 1 | 0 | 0 | 0 |
| F08 | 34 | 30 | 1 | 0 | 0 | 1 | 2 | 0 | 0 |
| F09 | 26 | 23 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| F10 | 34 | 27 | 3 | 0 | 0 | 1 | 3 | 0 | 0 |
| F11 | 21 | 15 | 5 | 0 | 0 | 0 | 1 | 0 | 0 |
| F12 | 25 | 22 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| F13 | 22 | 18 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| F14 | 12 | 11 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| F15 | 23 | 18 | 4 | 0 | 0 | 0 | 1 | 0 | 0 |
| F16 | 20 | 16 | 2 | 0 | 0 | 1 | 1 | 0 | 0 |
| F17 | 31 | 26 | 3 | 0 | 0 | 0 | 2 | 0 | 0 |
| F18 | 22 | 17 | 4 | 0 | 0 | 0 | 1 | 0 | 0 |
| F19 | 25 | 17 | 7 | 1 | 0 | 0 | 0 | 0 | 0 |
| F20 | 24 | 18 | 5 | 0 | 0 | 0 | 1 | 0 | 0 |
| F21 | 35 | 32 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| F22 | 29 | 25 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| F23 | 41 | 35 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| F24 | 26 | 23 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| F25 | 33 | 18 | 6 | 0 | 0 | 0 | 9 | 0 | 0 |
| F26 | 34 | 27 | 3 | 1 | 0 | 0 | 3 | 0 | 0 |
| F27 | 30 | 24 | 4 | 0 | 0 | 1 | 1 | 0 | 0 |
| F28 | 13 | 9 | 2 | 1 | 0 | 1 | 0 | 0 | 0 |

## Findings and proof

Four separately filed product defects, all in the authorized Canvas fork:

- [#124](https://github.com/enyst/playground/issues/124): an invalid draft retains stale Latest test passed status. Upstream copy18601 noted; no duplicate filing.
- [#125](https://github.com/enyst/playground/issues/125): automation detail controls clip at phone width, including RTL.
- [#126](https://github.com/enyst/playground/issues/126): built ESM Canvas library fails in a normal Vite host at requireReact.
- [#127](https://github.com/enyst/playground/issues/127): accepted plugin parameter edits are absent from actual creation. SDK 1.55.1 lacks runtime parameter support; merely forwarding a field is not claimed as the fix.

One unresolved failing check remains on F26.desktop-external-links even though that ID is classified existing-known from separate genuine #17112 paths. Electron external Markdown links remain an **unresolved integration candidate**, not a fifth filed defect: two real clicks failed to hand off, while the direct real OS browser control succeeded. Intake/root cause remains open. Existing SDK4554 critic behavior and SDK5622 cipher-mismatch preservation failures are existing-issue retests. Repeated checks are not additional issues; origins remain unconfirmed without comparable baseline evidence.

13 narrow recipes have verified two BASE xfail and two TARGET xpass attempts each: 26+26 attempts. [Scoped proofs](paired-proofs.json). Broader failures and unproved branches remain; selector comparison stays inconclusive because descriptors differ. A proof does not retire an entire issue or family.

## Runtime results and boundaries

Successful actual model IDs are `deepseek/deepseek-flash` and `deepseek/deepseek-v4-pro`; `deepseek-pro` is a profile label. Genuine model/tool evidence covers selected planning, tools, secrets, critic and compaction paths. Provider/Cloud/ACP branches without their exact prerequisites remain separate; no total provider bill is claimed.

A real Canvas-created Docker conversation completed a Flash call and one terminal action in its isolated container workspace. Host-mount/file/hostname correlation passed; deleting the conversation removed its container/workspace and the launcher returned to Local. A task-owned pinned minimal-image derivative added only public CA/proxy configuration. The stock image’s managed-proxy egress failure and the successful derivative are distinct outcomes. [Sanitized runtime proof](docker-runtime-proof.json). This does not pass the all-in-one Canvas image packaging test, browser/ACP capability matrix or ARM/Windows cases. The earlier postgres smoke is prerequisite-only and is excluded from mapped Canvas image observations. The original prompt-bearing screenshot remains private.

Helm 3.19.0 lint and default/Ingress template renders are valid static chart-artifact checks; no Kubernetes installation, token-readability or cluster proof is claimed. Live install needs an authorized disposable cluster/context. Actual TMUX explicit/default directory creation and frontend-only noncreation passed, with owned cleanup. Non-DNT consent used only an owned loopback discard sink: genuine opt-in, opt-out/re-opt-in within 3.563 seconds, final consent denied and owned cleanup. [Runtime checks](runtime-checks.json).

Real Electron/Xvfb failure splash was exercised. Playwright CDP clicked Copy, observed Copied after IPC completion, then Quit exited normally; clipboard contents were not read/exported, and owned display/process/private state was cleaned. External-link handoff is the separate unresolved candidate above, not a lack-of-controller limitation.

Verification-only harness corrections have 73 passing controller tests, syntax checks and live re-drives. Map checks retain 28 families/779 IDs and resolve 1110 cited test IDs. Product fixes and expected-behavior changes that approve bugs are excluded.

## Remaining work and handoff

**Blocked describes a prerequisite missing during this run.** Several checks can be enabled by supplying credentials or arranging a supported environment. Others need a specific compatibility state, historical evidence, or a change to the permitted test scope. They should be planned separately rather than treated as permanent limitations.

We tested the frozen synchronized Canvas target with its supported runtime pins (SDK 1.55.1, automation 1.20.1). The compatibility and exceptional-state rows below describe additional branches in the feature map; they do not mean those current versions failed to install or run. Existing DeepSeek credentials supported successful real model calls. Additional integration credentials/configurations were missing for the affected checks.

| Follow-up category | What was missing during this run | What would enable the check |
|---|---|---|
| **Credentials and account access the user can supply** | Authorized Cloud organizations/admin roles, connected Git providers, Slack, Linear/Atlassian OAuth, DigitalOcean and authenticated ACP workflows. | Identify each recipe’s exact secret names, endpoint/configuration and required account permissions; supply them through environment secret configuration or the service’s authorization flow, then rerun the affected paths. Credentials for one service do not establish the others. |
| **Genuine critic producer / service configuration** | `F06.critic-result` lacked a genuine detailed/positive critic producer or authorized Cloud critic configuration. The exact service credential names and configuration were not established in this audit. | Determine the supported critic service/producer and its configuration requirements, then supply any needed credentials and exercise an actual critic result. This is an actionable setup gap; credentials alone have not yet been demonstrated sufficient. |
| **Environment or equipment to arrange** | macOS titlebar/Gatekeeper and Windows upload-path environments, an accessible real microphone, and an authorized disposable Kubernetes cluster. | Provide the corresponding platform, device or cluster/context and run its specific recipe. Helm lint/template checks already passed; cluster installation remains separate. |
| **Writable Editor service installation path** | The official Editor dependency was prepared, but SDK 1.55.1 hardcodes `/openhands/.openvscode-server/bin/openvscode-server`; creating that host directory failed with **Errno 30**. | A supported runtime with that host-service path writable or a verified supported installation alternative. A Docker conversation workspace does not provide the global Editor service. |
| **Docker build capacity** | The all-in-one image build crossed the **5 GiB disk reserve** while unpacking its base, after more than 11.23 GB incremental consumption. | Enough supported Docker storage for the remaining unpack/build plus the reserve; the completed peak requirement is unmeasured. Frontend build and the separate real Docker conversation test already succeeded. |
| **Network/setup to resolve** | The combined six-server MCP configuration timed out during package fetch/revalidation; a Cloud integration page encountered browser TLS/trust failure. | Establish working package access/configuration within the combined runtime’s timeout, and working browser trust/access for the integration page. Isolated cached Time and HTTP successes do not prove the six-server combination. |
| **Compatibility or exceptional-state fixture** | Specific older compatible backends without particular APIs; absent usage metrics rather than zero-valued metrics; a genuine legacy/custom non-object Finish producer. The current SDK provides the APIs and requires object Tool responses. | Arrange the exact supported alternate backend or genuine producer and test the current Canvas consumer. These are separate compatibility/defensive-state checks, not ordinary current-stack startup requirements. |
| **A future update-available condition** | The installed release was current; the update-available branch needs a genuinely newer published release. | Retest when a newer release exists. The observed up-to-date branch does not prove the update-available UI; no fake registry result was counted as proof. |
| **Historical evidence to obtain** | Historical #17436 depends on reporter-private patched artifacts, their base identities and the underlying traceback, which were not supplied. | Obtain those exact artifacts/evidence for the historical comparison. The successful current stock-preset dispatch remains a separate control and cannot retire that historical report. |
| **Explicit execution boundary** | Some final confirmations unconditionally enable recurring jobs; those checks were excluded by the no-recurring-fixture boundary and prior approval rejection. | An explicitly permitted, confined test procedure with reliable owned cleanup would be needed before exercising those confirmations. No scope change is assumed here. Inert events and disabled drafts are distinct checks, not replacements. |
| **An existing product bug obstructing another entry point** | Existing #16792 prevented the phone Overview entry needed for a downstream check. | Verify a fix for that obstruction, then exercise the actual phone entry. Desktop and already-open-panel alternatives do not prove it. |

For the critic specifically, the actual patch-critic path reproduced existing SDK #4554. The built-in constant `PassCritic` demonstrated rendering only. Neither is proof of a genuine detailed/positive critic result; that missing producer/service setup is tracked separately from the existing product failure. No new critic credential names or service requirements are invented in this report.

These prerequisites cover **46 blocked-only IDs and additional branches in 19 otherwise-observed IDs**. The table groups follow-up work; it does not assign new mutually exclusive per-reason counts or change any recorded outcome. [Exact per-branch reasons and evidence references](remaining-blocked-subbranches.json) remain the detailed record. No follow-up test was run as part of this explanatory edit.

Other formerly missing fixtures were completed during the audit. The missing-manifest variant passed direct 404/sidebar absence and reproduced existing #18210 on desktop/phone Home links and the no-manifest toast. The serial packaging attempt is capacity-blocked: frontend build passed in 120.79 seconds, then full-base unpack was canceled at the reserve after 69.16 seconds; no final Canvas image/container was produced and no OOM was recorded. [Packaging receipt](docker-packaging.json). Main-window failure transition, standalone TMUX startup, cross-run terminal isolation and desktop/phone Home Docker Clear were exercised. Three real remote-navigation branches reproduced existing #17112; inert custom-scheme refusal is a limited observation. Exact early-stop late-ingress reproduction matches existing #16257 and supersedes its historical out-of-scope disposition without erasing that history.

All 23 CLI runs now have absent private directories and closed owned ports in the read-only snapshot. Retained artifacts include 1110 PNGs and 717 JSON files with readable hashes. Six additional manual runtime fixture receipts verify 27 file hashes, absent private paths, closed owned ports, no owned runtime processes, absent display, and removed owned temporary TMUX paths. Manual Electron cleanup has separate final receipts. Final Docker cleanup removed all three task images and exactly three new cache IDs while preserving all 34 preexisting cache IDs; only initial postgres and ryuk images remain. Read-only final cleanup found 23 CLI runs with absent private directories, closed owned ports, 1110 PNGs and 717 JSON files readable and 0 errors. Public artifacts exclude keys, session URLs, raw private traces, prompts and unnecessary runtime identifiers.

No next baseline is justified while required branches remain blocked. The final Home check is complete. [Draft maintenance PR #129](https://github.com/enyst/playground/pull/129) targets the Canvas fork main branch and remains open/unmerged. The final Home Recent activity consumer check also passed: three genuine custom-schema Finish callbacks (summary-only, custom-only, numeric-status) each verified on desktop and phone; status/summary/title fallback passed while existing phone clipping #18234 remains. Its owned fixtures and conversations were deleted and evidence retained. The user clarified that HUMAN requirements apply upstream only, outside this fork-only task. No HUMAN statement is filled. The dedicated fork tracker now carries ready-for-dev after an explicit repository-writer readiness assessment. No merge, recurring schedule or environment-template publication is part of this run. The coverage attempt report is complete with a blocked/partial verdict; publication follows the fork-only applicability described below. [Model-backed scope](model-backed-coverage.json) distinguishes credential availability, actual successful request model IDs and behavior evidence; private references are explicitly marked.

Latest not-run semantic reconciliation reviewed all 126 unchanged check labels: 105 historical generic placeholders and 21 scope-matched replacements. No constructible unfinished check remains in that set. Historical #17436 requires unavailable reporter-private comparison artifacts; non-object Finish requires a producer unsupported by the installed object-only Tool schema, and both have explicit current blocked entries. This does not make every recorded ID a complete branch pass.


## Final provenance, inventory and retention

The four source checkouts were confirmed as the enyst forks of Canvas (`playground`), software-agent-sdk (`agent-sdk`), extensions (`oh-extensions`) and automation (`automation`), each with default branch `main`. The comparison remains the frozen Canvas target above; maintenance commits do not replace the target product tree. SDK source was `5966c781ed014efc96c028a99bddd4ff78426741`, extensions `d05780951082d4d83f785cb1d4c7f2be6d1c14a5`, automation `bc4cca7c7e2eab5cb7d0ff671bec5dd113d3e2c2`. Installed runtime versions, rather than sibling source assumptions, were used for live claims. Canvas was 1.27.1; Node 24.21.0, Playwright 1.63.0 and Chromium 151.0.7922.173 drove the shell browser harness on Debian 13.

The frozen existing-issue inventory contains 383 non-PR issues. Current dispositions are 291 mapped, 17 historical not-reproduced, and 75 out of scope. These dispositions are not 291 fresh passes. Current live proof for existing #16257 supersedes its historical out-of-scope disposition without erasing the historical evidence. Thirty changed commits were reconciled to thirty PR descriptions, reviews and files plus 27 linked issue bodies; all 122 changed paths were accounted for. Closed/updated query counts were not treated as fixed-behavior counts.

The final root readback at 22:18:46 UTC found no audit-owned runtime processes and 16,435,482,624 free disk bytes. The post-stop evidence manifest hashes 2111 retained regular files totaling 115,184,960 bytes, including 1110 PNGs, 717 JSON files, six MP4s and two GIFs. These are retained-evidence counts, not public-release or test counts. Sanitized private evidence remains protected in the original audit workspace; private run credential state is purged. Public artifacts are a deliberately reviewed subset. No shared data was pruned.

Infrastructure interruptions remain separate from product results. The original shared execution-channel loss interrupted the first turn; surviving services did not complete the audit independently. A later brief executor-link interruption at 21:53:38.855 UTC reconnected at 21:53:45.168 UTC while desktop API reads continued. Neither event establishes a resource cause, host migration or product failure. The serial packaging cancellation, in contrast, has its own measured disk-reserve receipt.

Private retained evidence remains in the original `/workspace/work/weekly-20261017/`: `runs/*/evidence/ledger.jsonl` and adjacent screenshots/JSON/video; `aggregate-coverage.json` for the per-ID join; `source-F01.md` through `source-F28.md` for source review; `proof-summary.json` for paired attempts; `inventory-review.json` and `intent-runtime-join.json` for issue/PR reconciliation; `all-evidence-files-final-sha256.json` and `cleanup-retention-final.json` for final retention/cleanup. These are evidence locations, not links to a publicly released raw bundle. Original chronology and failed setup receipts are preserved.

## Maintenance PR preparation and fork policy

Published draft PR [#129](https://github.com/enyst/playground/pull/129), title: **docs(verify-openhands): reconcile weekly map and prove controller corrections**.

[Fork maintenance tracker #128](https://github.com/enyst/playground/issues/128) has concrete acceptance criteria and now carries `ready-for-dev`. This is a dedicated maintenance tracker, not a fifth product defect. `.github/workflows/issue-readiness-check.yml` expressly allows repository writers to grant readiness; the fork writer capability was verified. This deliberate publication-readiness decision does not relabel product issues as a side effect of map bookkeeping.

The user explicitly clarified applicability: AI attribution goes first in fork posts; HUMAN requirements are upstream-only and outside this fork-only audit. The literal first line remains `I am an AI agent helping Engel Nyst (@enyst) with project work.` The copied template HUMAN segment remains untouched and empty; no human test statement is invented. No upstream validator or product file was changed.

For provenance, `AGENTS.md:49` reserves HUMAN for people; the template says “Human contributors: add a short note about your testing.” The inherited validator requires HUMAN first and 20 visible characters of human prose. Local validation therefore reports those two inherited upstream-policy errors; they are not represented as passing. Under the direct user clarification they do not block this fork publication. The independent authenticated linked-issue readiness validator passed against the fork tracker. The full proposed body and validation receipts accompany this report.

Recorded blocked-key reconciliation: of 98 chronological blocked keys, seven fixture rationales were superseded by actual later proof and five old prerequisite rationales were replaced by current blocks. Eighty-five retain current recorded prerequisite reasons and one is blocked by existing #16792. The compact remaining list spans 65 feature groups: 46 blocked-only IDs and 19 IDs with other observed branches. These are grouped prerequisites, not 65 independent new defects. [Exact unexercised subbranches and missing inputs](remaining-blocked-subbranches.json). No unresolved semantic branch classification remains; the separate executed Markdown handoff failure still has unresolved cause.

Publication readback: the draft PR is open/unmerged with the exact prepared AI-attributed body. The actual fork “Validate PR description” CI job failed on the same two inherited HUMAN-first/nonempty-HUMAN checks; [sanitized actual job receipt](pr-ci-description-check.json). The user’s applicability clarification permits fork publication, but does not rewrite that inherited workflow or turn its result green. Skill-isolation, artifact and title checks passed in the initial readback; broader CI jobs were still running and are not substituted for live audit proof.
