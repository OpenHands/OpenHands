I am an AI agent helping Engel Nyst (@enyst) with project work.

<!-- Keep this PR as draft until it is ready for review. -->

HUMAN:

<!-- Human contributors: add a short note about your testing. -->

---

AGENT:

<!-- AI/LLM agents:
Do not edit the HUMAN section.
Write a concise summary of what changed and link any reviewer artifacts, such
as files under `.pr/`. For HTML artifacts, include a rendered preview link:
https://htmlpreview.github.io/?https://github.com/<owner>/<repo>/blob/<commit-sha>/.pr/<file>.html
In this AGENT section and the template fields below, provide evidence that the
code runs properly end-to-end. Just running unit tests is NOT sufficient. Explain
exactly what command you ran and include logs, screenshots, or reproduction notes.
-->

## Why

The verification map and controller need corrections proved during the full weekly maintenance audit at frozen Canvas target 61a3bd93e642a0ff4def73b29496a73245628209. Product behavior and the accepted baseline remain unchanged.

## Summary

- Keep narrow, repeatably verified fix recipes and current known failures accurate.
- Correct controller cleanup, redacted request projection and browser media controls with live evidence.
- Retain a scoped maintenance report and reviewed evidence; no product fixes.

## Issue Number
<!-- Required. The linked issue must carry the `ready-for-dev` label, which
means it has clear acceptance criteria (and, for bugs, reproduction evidence).
If no such issue exists yet, open one using the Bug or Feature Request template
and wait for it to be labeled `ready-for-dev` before opening this PR. -->
Fixes #128

## How to Test

From the Canvas checkout, use Node 24+ and run:

```sh
.agents/skills/verify-openhands/scripts/control-openhands map check
.agents/skills/verify-openhands/scripts/control-openhands map coverage
.agents/skills/verify-openhands/scripts/control-openhands map testids
```

All passed: 28 families, 779 stable IDs, zero unmapped routes/component directories, and 1110 resolved cited test IDs. The controller Vitest suite passed 73 tests; syntax and diff checks passed. Live re-drives cover redacted request projection, media preferences, owned process-group cleanup and the checked-in app-backend fixture. See the final report and neighboring proof receipts for reproducible steps and screenshots.

Full weekly source review covered 28/28 families; live observations touched 28/28 families. Mutually exclusive IDs: 633 pass-only recorded observations, 88 existing-known failures, 4 newly filed failures, 8 fix assertions, 46 blocked-only; total 779. Observed is not all-branches-pass. The 13 paired fix recipes each have two baseline xfail and two target xpass attempts, separate from whole-map coverage. Required unavailable branches remain blocked; the accepted baseline is unchanged. All owned CLI runs are stopped and private run state purged; retained evidence hashes remain readable.

<!--
Required. Share the steps for the reviewer to be able to test your PR. e.g. You can test by running `npm install` then `npm build dev`.

For bugs, include reproduction steps and observed before/after results.
For functional changes, logs and tests supplement the running-Canvas demonstration below.
Nonvisual evidence alone is sufficient only for non-functional changes.

If you could not test this, say why.
-->

## Video/Screenshots

[Reviewed final report and evidence](https://github.com/enyst/playground/blob/c461066866f4dcffec9e845995b57e9ba25ffb89/.agents/skills/verify-openhands/references/backfills/evidence/2026-10-17/final/weekly-report.md). [Live controller receipts](https://github.com/enyst/playground/blob/c461066866f4dcffec9e845995b57e9ba25ffb89/.agents/skills/verify-openhands/references/backfills/evidence/2026-10-17/final/controller-live-proof.json) and [running Canvas fixture](https://github.com/enyst/playground/blob/c461066866f4dcffec9e845995b57e9ba25ffb89/.agents/skills/verify-openhands/references/backfills/evidence/2026-10-17/final/checkedin-app-backend-fixture.png) supplement the paired proof results. Prompt-bearing Docker/model screenshots remain private. The real minimal-image Docker conversation succeeded; full packaging remains capacity-blocked after frontend build success.

<!--
Functional changes require screenshots/video of the running Canvas exercising
the changed behavior, even outside frontend files. For bugs, demonstrate failure
before and success after using the same setup. For new features, show the working
behavior. Use video for timing or transitions. Logs/tests alone do not suffice.
Non-functional changes may use relevant commands and results under How to Test.
-->

## Design Doc

<!--
Optional, encouraged for non-trivial PRs. Add a self-contained HTML design doc under the
temporary `.pr/` directory (e.g. `.pr/design.html`) covering the code/API design and a
before/after of your change, then link it here via htmlpreview at the commit that contains it:

  https://htmlpreview.github.io/?https://github.com/<your-fork>/<repo>/blob/<commit-sha>/.pr/design.html

Use the commit SHA, not the branch name: `.pr/` is removed automatically through
a cleanup PR after this PR merges to `main`, and branch links may stop working.
See docs/DEVELOPMENT.md ("Design doc for non-trivial PRs") for details.
-->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Breaking change
- [x] Docs / chore

## Notes

<!-- Optional: migrations, config changes, rollout concerns, follow-ups, or anything reviewers should know. -->

Fork publication note: the user explicitly clarified HUMAN requirements as upstream-only. This fork PR preserves the empty HUMAN template segment and starts with AI attribution. Inherited local HUMAN-first/nonempty-HUMAN errors remain recorded; the fork readiness gate is checked separately.
