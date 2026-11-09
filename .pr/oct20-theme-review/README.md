# October 20 review follow-up — PR 17730

Reviewed the current PR, requests, discussions, and reviews as authenticated
`rajshah4` before changing the clean isolated checkout.

Fresh findings:

- https://github.com/OpenHands/OpenHands/pull/17730#discussion_r4324530249
- https://github.com/OpenHands/OpenHands/pull/17730#discussion_r4324530257

The October 5 shared focus fix and October 19 ACP focus fix were already
addressed. They were preserved and exercised again, rather than treated as new
regressions.

## Changes

Merged main `3bf5e85c01dce3cc1a8ebdfce5cd801bd133dd27` normally in `9e1e88bfa`.
Resolved the four chat-icon conflicts by keeping main's `currentColor`
inheritance through contrast-aware wrappers, while retaining this PR's light
control borders. The browser regression verifies branch, repository, external
link, and workspace icons in all five themes, including a custom host override
of `--oh-contrast`.

The draft badge now uses the opaque warning badge tokens, with its former
15% warning fill and warning ink retained as dark overrides. The improved guard
recognizes both named and arbitrary `var()` utilities, including mixed forms,
and ignores variant-only compatibility overrides. It first failed on the
reported draft badge and on two equivalent setup states. Those setup success
and missing-draft notice states now use the same shared badge/banner tokens,
with their previous dark styling retained.

The rendered-component contrast regression first failed below 4.5:1 on raised
Light+ and Solarized panel surfaces, then passed after the fix. The shared test
color resolver now also measures arbitrary CSS-variable utilities, so the old
badge cannot evade the actual contrast calculation.

## Real application captures

Before: detached checkout at `adf16c26b`, origin `http://127.0.0.1:33770`.
After: merged checkout plus the review fixes, origin `http://127.0.0.1:34770`.
Both used the production frontend, Agent Server **1.56.0**, Automation **1.22.0**,
private state and Chromium through `control-openhands`. No MSW or model execution
was used for these captures. Health checks passed. The after build was made
before the merge commit was recorded; the verification tool subsequently
reported a build-input fingerprint warning from that commit transition. The
product source did not change between the build and these captures.

Fixture arrangement: an inert event automation (`qa-theme` / `qa.never`) was
created disabled through the private backend API. Only that fixture's private
SQLite row was assigned the persisted DRAFT state to view the real detail-header
badge without dispatching a test run. This is rendering evidence, not proof of
the draft materialization lifecycle. The ready-to-test draft was saved through
the normal setup UI. The missing-tagged-draft notice is covered by unit tests;
the direct missing-draft URL takes a different route and was not used as live
proof of that notice.

The real header capture demonstrates the changed fill. The below-4.5:1
regression on other panel hosts is demonstrated by the rendered-component
contrast test, rather than claimed for the header's plain page background.

| Capture               | Before                                    | After                                   |
| --------------------- | ----------------------------------------- | --------------------------------------- |
| Light+ draft          | [Before](before-light-draft.png)          | [After](after-light-draft.png)          |
| Solarized draft       | [Before](before-solarized-draft.png)      | [After](after-solarized-draft.png)      |
| Neutral draft         | [Before](before-dark-draft.png)           | [After](after-dark-draft.png)           |
| Light+ setup validity | [Before](before-light-setup-validity.png) | [After](after-light-setup-validity.png) |

Merged icons in the real private Git workspace: [Neutral](after-dark-icons.png),
[Light+](after-light-icons.png), [Solarized Light](after-solarized-icons.png).
The fixture repository has an inert example origin; no push or conversation
execution was performed.

## Verification

- Focused unit suites: **530 passed**, 42 files.
- Full suite on Node 24 / UTC: **9,399 passed**, 7 todo; only the two unchanged
  macOS Bash 3.2 Docker-policy tests failed locally.
- **15 browser/theme combinations verified** across Chromium, Firefox and
  WebKit: existing shared fields, ACP keyboard focus/blur, rendered draft
  contrast, four icon consumers and host contrast override. Two Firefox cases
  were rerun successfully after build-time dev-server reloads disrupted the
  initial sweep (13 passed initially, both affected cases passed on rerun).
- `npm run lint`: passed, zero errors and existing warnings.
- Production app build and `npm run build:lib`: passed.
- `git diff --check`: passed.

No PR description/HUMAN section, review state, branch protection, or other
repository was changed. Maintainer review remains separate from this update.
