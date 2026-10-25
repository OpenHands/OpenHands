# Issue #18690 verification

Captured from the real local Agent Canvas stack on macOS arm64. Agent Server
1.56.0 and automation 1.22.0 were used before and after the change. The baseline
was OpenHands/OpenHands commit 8d39305538c1d17a30cbe477e57a0e18fcee7d95.
The local executable plugin is a verification fixture, not a mocked API response.
All automations were disabled. No run was dispatched and no LLM was called.

## Reproduction

1. Use Node 24 and install the committed dependency lock with `npm ci --ignore-scripts`.
2. Launch the repository's `control-openhands` CLI with `launch --new`, then run
   `doctor` and `onboard --skip` against that isolated run.
3. Create a local plugin with `.plugin/plugin.json` containing a name, version,
   description and an `automation` object: schema version
   `openhands.automation.plugin.v1`, script entrypoint `python3 worker.py`,
   working directory `scripts`, and an object config schema.
4. Create the inactive automation using `api POST /api/automation/v1/plugin-spec
   --write --data @<request.json>`. The request supplies the local plugin source,
   an event trigger, timeout 300, and config containing a long instruction,
   `dry_run: false`, `retry_count: 0`, and `routing: {queue: "reviews"}`.
5. Open `/automations/<response-id>` through `browser goto`.
6. On the baseline the entrypoint is followed by the generic unavailable-tarball
   message and the plugin config/timeout are missing (`before.png`).
7. On this branch the runtime explanation, source, config and timeout are visible
   (`after.png`). Show more/less reveals and collapses the complete instruction
   (`expand-collapse.gif`). Phone width was 390 px (`phone.png`), with no
   document-width overflow.
8. A second real request with an empty config and optional `ref`/`repo_path`
   verified that source metadata renders without an empty config block
   (`empty-config.png`). A local source does not check out the supplied ref;
   this check concerns the response's source metadata presentation.

## Additional checks

- A real uploaded tarball still lists `main.py` and its `print("pong")` contents.
- A real prompt preset still renders its prompt. The existing Export, Tarball,
  Edit, Turn on/off and Delete actions remain present.
- Plugin detail navigation requests the existing detail and runs endpoints;
  no tarball or plugin-definition fetch is made. No app/page errors observed.
- The six added regressions failed against the baseline; the related suite then
  passed all 36 tests after the fix.
- `npm test`: 845 files passed, 9250 tests passed, 7 existing todo.
- `npm run lint`, `npm run build`, `npm run build:lib`, and
  `npm run check-translation-completeness` passed.
- Full tests used Node 24.22.0 and Python 3.13 with native process permissions.
  An initial sandboxed run hit unrelated Python 3.9 and launcher-permission
  failures; the complete native run passed. Test-created user state was cleaned.

## PR #18700 review follow-up (2026-10-25)

The review identified a remaining unsupported action: runtime plugin bundles
skipped file listing, but their menu still offered Tarball, whose endpoint
returns HTTP 422. The menu now uses the same shared
`PLUGIN_SPEC_RUNTIME_TARBALL_PATH` sentinel as file listing. Other tarball paths
retain their download action.

### Comparison and scope

The baseline was `deced1d448237eaa52a2bf87d0dcff4d760fb94a` (build fingerprint
`02590d22592a20aa`). The target was that revision plus the product patch committed
with this record (served fingerprint `6b52b2c3c2f53100-dirty`). Both builds used
Agent Server/SDK **1.58.1** and automation **1.25.1**, on macOS 26.4 arm64,
Chrome 155.0.8059.86, Node 24.22.1 and Python 3.13. The app used its default
theme/fonts. The isolated runs were `pr18700-before-20261025` and
`pr18700-after-20261025`, from 09:24 to 09:37 UTC on 2026-10-25. Local ingress,
Agent Server, automation, frontend and editor ports were 18800, 18801, 18802,
18803 and 19801 respectively.

Launch each revision with `control-openhands launch --new --build always`, run
`doctor` and `onboard --skip`, then create an inactive local plugin-spec
automation through `POST /api/automation/v1/plugin-spec`. The fixture declared a
script entrypoint `python3 worker.py` under `scripts`, an object config schema,
config `{ "channel": "engineering-updates" }`, timeout 300, and a disabled UTC
January 1 cron trigger. Open its detail page and the Automation actions menu.
Both runs used the same fixture and backend versions. No automation was enabled
or dispatched, and no LLM was called.

| Check / viewport | Baseline observation | Target observation | Evidence |
| --- | --- | --- | --- |
| Runtime plugin menu / 1440×1000 | Tarball visible; clicking it returned HTTP 422 | Tarball absent; Export, Edit, Turn on and Delete retained; no tarball request on navigation | [Before](runtime-bundle-before-desktop.png), [after](runtime-bundle-after-desktop.png) |
| Runtime plugin menu / 390×844 | Tarball visible | Tarball absent; remaining actions visible | [Before](runtime-bundle-before-phone.png), [after](runtime-bundle-after-phone.png) |
| Uploaded script / 1440×1000 | Existing behavior covered by the original verification above | Real upload rendered `main.py`; Tarball returned HTTP 200 and downloaded a gzip archive containing `main.py` | [Uploaded bundle](uploaded-bundle-download.png) |

All three targeted live checks passed. This is manual review-follow-up evidence,
not a full automation-family sweep or a registered known-bug proof. The target
reported zero app/page errors. A repeat check initially used an incorrect
Automation actions selector; it was corrected using the observed accessible
name and repeated successfully. The verification specification and harness were
not changed.

### Regression and build results

- The menu regression first failed against the baseline: 1 failed, 15 passed.
- The related automation-detail suite passed: 16 files, 154 tests.
- Full `npm test`: **854 files passed, 9486 tests passed, 7 existing todo**.
- `npm run lint`: passed, with 614 existing warnings and zero errors; TypeScript
  and formatting checks passed.
- `npm run check-translation-completeness`: passed.
- The standalone build used by the target live launch and `npm run build:lib`
  both passed. Dependencies came from `npm ci --ignore-scripts` with the
  committed lockfile; no dependency or locale changes were made.

Both owned stacks and their browser processes were stopped. The target run's
private state and inactive fixtures were purged; only reviewed screenshots were
copied here. Evidence was retained and checked after shutdown. No verification
baseline was promoted.

This report was generated by an AI agent (Codex) on behalf of the requester.
