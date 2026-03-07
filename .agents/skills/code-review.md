---
name: code-review
description: OpenHands repository-specific PR review guidance for automated reviewers.
triggers:
  - /codereview
---

# OpenHands PR Review Guidance

Use this guidance when reviewing pull requests in `OpenHands/OpenHands`.

## Prioritize high-signal findings

1. **Behavior regressions**
   - Focus on logic or state-flow changes that can break runtime behavior.
   - Flag breaking changes to user-facing APIs, SDK behavior, or event contracts.

2. **Security boundaries**
   - Be strict around shell execution, sandbox/runtime boundaries, auth/session handling, and secret exposure.
   - Flag any path traversal, command-injection, or unsafe deserialization risks.

3. **Cross-surface consistency**
   - Validate backend/frontend/API contract alignment (types, response shapes, error handling).
   - Ensure docs and examples match real behavior for changed features.

4. **Tests and verification**
   - Recommend targeted tests for changed code paths (not blanket “add tests” comments).
   - Prefer precise missing-coverage comments with concrete scenarios.

## Reduce low-value noise

- Avoid style-only comments if existing lint/format tooling already enforces the rule.
- Avoid speculative architecture rewrites for small bugfix PRs.
- Skip comments that restate obvious code without actionable change.
- Do not block on external infra flakes unrelated to the PR diff.

## Comment style

- Be concise and specific.
- Include exact file/line references.
- Explain user impact and propose a concrete fix.
- Prefer one high-confidence actionable comment over many weak comments.
