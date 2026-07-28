<!--
Append this block to each production repo's .github/PULL_REQUEST_TEMPLATE.md.
The landing-checklist GitHub Action parses this exact block (matched via the
"<!-- landing-checklist:v1 -->" marker and the item labels below) — do not
reword the checklist item text without also updating
landing-checklist-reusable.yml.

Only fill this in if your PR title starts with "(feat)". Other PR types can
delete this section.
-->

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
