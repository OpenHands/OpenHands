# Issue #17891: review update, November 7

The review fixes are implemented in `320e309678ba9745d8d786f8baffe4d38d500fc4`, merged with upstream `c90c6199dcf57e9208ad2adfac2aef5b41e04ec8`. [Design and pinned source diffs](design.html).

## Review changes

- `useAppTitle` reads matched route handles rather than URL wildcards. `/settings/does-not-exist`, `/settings/llm/deeper` and `/automations/a/b` display the not-found page with the `OpenHands` title.
- Both `/` and `/conversations` use the existing localized New Chat label. Settings children inherit the matched parent title. Customize, MCP, Skills, Plugins and Apps declare their existing page translations. Existing extension contribution titles, including nested extension paths, are preserved from current upstream.
- Automation titles use `getInterfaceCopy().sidebarLabel` only when `hasAutomationInterface()` admits the interface. The default live stack displays `Automate | OpenHands`. Unavailable interfaces retain the app fallback.
- Two distinct shortcuts remain: New conversation and Automations. Manifest-relative targets and explicit `scope: "."` keep both inside effective scope at `/` and `/canvas/`, including installation from a nested Settings page. App names, branded icons and the manifest link remain with prerequisite #17895.
- `navigate-existing` intentionally navigates the existing window with a full document reload. Unsaved Settings or automation form input can be lost. No navigation blocker or client-side launch consumer is added; browser support and user capture settings still apply.
- Manifest fixtures are anchored to `__dirname`; running the manifest test from the parent directory also passes.

## Tests and builds

Node 24.15.0, macOS arm64, locked dependencies via `npm ci --ignore-scripts`.

- [42 focused tests pass](review-focused-tests.log): real data router and route exports, real translations/query cache/status store, existing extension runtime tests, and manifest scope/route/icon contracts. Only the underlying conversation service is mocked; interface admission/copy accessors are controlled for their edge case.
- Before implementation, the new title regressions produced 12 failures. The two-shortcut contract failed against the old manifest; after removing its duplicate, the nested-install scope tests still failed until explicit scope was added.
- [Manifest tests from parent working directory: 3 pass](review-parent-cwd.log).
- [`npm run lint`: zero errors, 183 warnings](review-lint.log). `npm run build` and `npm run build:lib` pass. [Translation completeness passes](review-translations.log).
- [`TZ=UTC npm test -- --maxWorkers=2`](review-full-tests.log): 886 files pass, 2 fail; 10,282 tests pass, 3 fail, 1 skipped, 7 todo; 424.30 seconds. The failing tests and their production tooling have no diff against current upstream. Python 3.9 cannot import `datetime.UTC`; [the same three-test file passes under Python 3.12](review-python312.log). Two Docker session-key tests encounter Bash 3.2's empty-array `set -u` behavior; [direct reproduction](review-bash32.log) exits 127 with `unbound variable`. These unrelated files were left unchanged.

## Real browser before and after

Two isolated Local stacks used the existing read-only `control-openhands` driver, the original review head `56425e4a17a3e914770c41081cfedea7731c14ce` and the fixed implementation above. Both ran Agent Server 1.60.1 and automation 1.30.1 in Chrome 155. `doctor` passed; onboarding was skipped and analytics disabled. No API key or model call was used. [Baseline health](review-baseline-health.json), [fixed health](review-fixed-health.json).

Commands: `launch --new --run-id pr18635-review-{baseline,fixed} --sdk-version 1.60.1 --automation-version 1.30.1 --min-free-mb 0`, `doctor`, `onboard --skip`, then `browser record start`, `browser goto <route>`, `browser eval '({title:document.title,path:location.pathname,...})'`, `browser screenshot`, and `browser record stop`, always with the owned `--run` directory.

| Page | Before | After |
|---|---|---|
| Three invalid nested URLs above | Settings / Automations title on 404 | OpenHands |
| `/` and `/conversations` | Home / Conversations | New Chat / New Chat |
| `/automations` | Automations | Automate (interface sidebar label) |
| `/mcp` | OpenHands | Model Context Protocol (MCP) |
| `/apps` | OpenHands | Apps for Agent Canvas |

[Before recording](review-baseline.webm), [after recording](review-fixed.webm), [before readbacks](review-baseline-readbacks.json), [after readbacks](review-fixed-readbacks.json), [before 404](review-baseline-invalid-route.png), [after 404](review-fixed-invalid-route.png). Recordings capture page content rather than native browser chrome; readbacks establish actual document titles. The baseline UI advertises a newer version, reflecting its older frontend. Both stacks use the same backend pins.

A second [sidebar recording](review-sidebar.webm) uses actual Apps → Automate → New Chat → Customize navigation. [Readbacks](review-sidebar-readbacks.json) show the expected titles, identical `performance.timeOrigin` across transitions, and zero page/app errors or warnings. These are browser-tab checks, not a new installed-window test.

## Pending prerequisites and independent verification

Keep the PR draft until PWA foundation #17895 lands, then integrate its installation metadata and capture the installed-window transitions requested by #17891. #17895 is still open and draft. The [historical October 18 integration report](historical-october-18.md) preserves the human's manual installation/window-reuse check against temporary prerequisite metadata; its three-entry Dock screenshot predates removal of the duplicate shortcut. It does not verify installation from this branch or the final two-entry menu. Installed behavior under a deployment base path remains unverified.

F01.document-title still expects generic `OpenHands` on the composer and after leaving an extension page. Track updating that expectation in a separate verification-skill PR, after the product change lands and in a separate agent context. No verification-skill files are changed by this PR relative to upstream. The manifest has no owning map entry; report that coverage gap rather than changing the verification specification here.

The linked issue is `ready-for-dev` but lacks the `enhancement` label required by the Feature PR-description check. A maintainer must add the issue type label. This PR stays correctly classified as Feature.

The standalone root layout retains the sole document-title writer. Conversation title/status logic and shared backend contracts are unchanged; no compatibility-floor update is needed. Local/Cloud, OpenHands/ACP and standard/planning/delegated conversations continue through that existing conversation branch. Embedded exports do not mount the standalone title writer.
