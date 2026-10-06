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

<!-- Describe problem, motivation, etc. -->

## Summary

<!-- 1-3 bullets describing what changed. For bugs, state the failure and fix. -->
-

## Issue Number
<!-- Required. The linked issue must carry the `ready-for-dev` label, which
means it has clear acceptance criteria (and, for bugs, reproduction evidence).
If no such issue exists yet, open one using the Bug or Feature Request template
and wait for it to be labeled `ready-for-dev` before opening this PR. -->
Fixes #

## How to Test

<!--
Required. Share the steps for the reviewer to be able to test your PR. e.g. You can test by running `npm install` then `npm build dev`.

For bugs, include reproduction steps and observed before/after results.
For functional changes, logs and tests supplement the running-Canvas demonstration below.
Nonvisual evidence alone is sufficient only for non-functional changes.

If you could not test this, say why.
-->

## Video/Screenshots

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

Use the commit SHA, not the branch name: `.pr/` is removed from the branch when a
same-repository PR is approved, and a branch link stops working then.
See docs/DEVELOPMENT.md ("Design doc for non-trivial PRs") for details.
-->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Breaking change
- [ ] Docs / chore

## Notes

<!-- Optional: migrations, config changes, rollout concerns, follow-ups, or anything reviewers should know. -->
