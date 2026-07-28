<!-- Keep this PR as draft until it is ready for review. -->

HUMAN:

<!-- Human contributors: add a short note about your testing. -->

AGENT:

<!-- AI/LLM agents:
Do not edit the HUMAN section.
In this AGENT section and the template fields below, provide evidence that the
code runs properly end-to-end. Just running unit tests is NOT sufficient. Explain
exactly what command you ran and include logs, screenshots, or reproduction notes.
-->

---

## Why

<!-- Describe problem, motivation, etc. -->

## Summary

<!-- 1-3 bullets describing what changed. -->
-

## Issue Number
<!-- Required if there is a relevant issue to this PR. -->

## How to Test

<!--
Required. Share the steps for the reviewer to be able to test your PR. e.g. You can test by running `npm install` then `npm build dev`.

If you could not test this, say why.
-->

## Video/Screenshots

<!--
Provide a video or screenshots of testing your PR. e.g. you added a new feature to the gui, show us the video of you testing it successfully.

-->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Breaking change
- [ ] Docs / chore

## Notes

<!-- Optional: migrations, config changes, rollout concerns, follow-ups, or anything reviewers should know. -->

<!-- landing-checklist:v1 -->
### 🚀 Feature Landing Checklist
_Required before merge:_
- [ ] Documentation added for the feature on the website
  _(add the page to `OpenHands/docs`, but set `hidden: true` in its frontmatter — the feature isn't public yet. Automation removes `hidden: true` only after two tech-council approvals and independent verification that the flag is enabled in production.)_
- [ ] E2E test added covering regression / up-to-spec behavior
- [ ] Feature is gated by an ENABLE_<FEATURE> flag and exposed through the appropriate Helm / embedded-cluster configuration

_Tracked post-merge (do not check manually — automation updates these):_
- [ ] Bug bash completed (3+ engineers), issues filed for next cycle
- [ ] Feature is available in the most recent release
- [ ] Feature included in an X / LinkedIn post

Linear tracking ticket: _(auto-linked by automation once PR is opened)_
Feature flag name: `___________`
