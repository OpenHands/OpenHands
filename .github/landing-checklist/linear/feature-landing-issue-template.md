# Feature Landing

Use this body for the shared Linear `Feature Launches` issue template. Automation
1 fills the metadata fields and copies the six checklist lines exactly.

## Source

- Feature PR: TBD
- Repository: TBD
- PR author / landing DRI: TBD
- Owning Linear team: TBD
- Merge commit: TBD
- Feature flag: TBD
- Docs page: TBD
- Social post: TBD

## Landing checklist

- [ ] Documentation added for the feature on the website
- [ ] E2E test added covering regression / up-to-spec behavior
- [ ] Feature is gated by an ENABLE_<FEATURE> flag and exposed through the appropriate Helm / embedded-cluster configuration
- [ ] Bug bash completed (3+ engineers), issues filed for next cycle
- [ ] Feature is available in the most recent release
- [ ] Feature included in an X / LinkedIn post

## Evidence contract

- Docs: merged `OpenHands/docs` PR and `.mdx` path with `hidden: true` until
  independently verified flag-on.
- E2E: test file and scenario that exercise the feature behavior.
- Self-hosted: exact flag plus implementation and Helm / Replicated paths or
  linked downstream PRs (code wiring, verified at review time).
- Bug bash: `bug-bash-report: <date> | attendees: <three or more distinct
  Linear users> | issues: <zero or child issue IDs> | helm-test: <note or
  link confirming a real Helm-based install test with the flag ON> |
  replicated-test: <note or link confirming a real Replicated/embedded-cluster
  install test with the flag ON, or n/a-pending-support if repos.yml's
  capabilities.replicated_preview_supported is still false>`. The
  `helm-test` field is always required; `replicated-test` becomes required
  (not just `n/a-pending-support`) once that capability flag flips to true.
- Release: release tag, image, chart, and production pin as applicable to the
  repository's rule in `repos.yml`.
- Social: public `x.com`, `twitter.com`, or `linkedin.com` post URL.

Do not move the feature to council review until items 1-5 have evidence. Do not
move it to `stage:ga` until item 6 has evidence and the feature flag is removed.
