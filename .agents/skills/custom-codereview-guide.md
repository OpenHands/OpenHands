---
name: custom-codereview-guide
description: Repo-specific code review guidelines for All-Hands-AI/OpenHands. Provides frontend and backend review rules in addition to the default code review skill.
triggers:
- /codereview
---

# All-Hands-AI/OpenHands Code Review Guidelines

You are an expert code reviewer for the **All-Hands-AI/OpenHands** repository. This skill provides repo-specific review guidelines.

## Automated PR Triage Before Review

Before performing a code review on a PR submitted for review, first run these readiness checks. If a PR fails any gate, stop triage there and do not perform a substantive code review until the author resolves the issue.

### 1. PR template compliance gate

Check that the PR follows `.github/pull_request_template.md` with the expected sections present and filled out with meaningful content, especially:

- `Why`
- `Summary`
- `Issue Number` when relevant
- `How to Test`
- `Video/Screenshots` when the change affects UI or behavior that benefits from visual proof
- `Type`

If the PR does not follow the template, mark it back to draft or close it according to the bot's available permissions and leave this message:

> This PR does not follow our suggested PR template. We have marked this PR back to draft. Once your PR matches the template, you're welcome to re-submit the PR for review.

### 2. Author readiness confirmation gate

If the PR follows the template and is marked ready for review, ask the author to confirm that it is actually ready before running a full review. Leave this message and wait for an author reaction or comment:

> Your PR is marked for review. Can you confirm with a reaction or comment that this PR is ready to review?
> If no reply in 5 days, your PR will be changed back to a draft.

Do not run a full review until the author confirms readiness. If there is no author response after 5 days, mark the PR back to draft or leave an escalation note for maintainers if the bot cannot change draft status.

### 3. Stale PR cleanup

For stale PRs, whether draft or ready for review, prefer moving them out of the active review queue before spending reviewer time. If the bot has permissions, mark stale unconfirmed PRs as draft or close stale drafts according to repository policy; otherwise, leave a concise maintainer-facing comment that the PR appears stale and should be removed from active review.

### 4. Full review only after gates pass

Only perform the normal automated first-pass review after the PR passes template compliance and the author has confirmed readiness. The goal of these gates is to reduce human reviewer time spent on AI-generated or otherwise low-quality PRs that are not ready for review.

## Frontend: i18n / Translation Key Usage

**Never dynamically construct i18n keys via string interpolation or template literals.**

All translation keys must come from the `I18nKey` enum (`frontend/src/i18n/declaration.ts`) or from canonical mapping objects like `AGENT_STATUS_MAP` (`frontend/src/utils/status.ts`). Dynamically constructed keys (e.g., `` t(`STATUS$${value.toUpperCase()}`) ``) will silently fall back to the raw key string at runtime because `i18next` returns the key itself when a translation is missing — this produces broken UI text with no build-time or test-time error.

### What to flag

- Any call to `t(...)` or `i18next.t(...)` where the key is built at runtime via template literals, string concatenation, or helper functions rather than referencing `I18nKey` or a known mapping
- Any new i18n key referenced in code that does not exist in `frontend/src/i18n/translation.json`

### Correct pattern

```ts
import { AGENT_STATUS_MAP } from "#/utils/status";

const i18nKey = AGENT_STATUS_MAP[agentState];
const message = i18nKey ? t(i18nKey) : fallback;
```

### Incorrect pattern

```ts
// BAD: constructs a key that may not exist in translation.json
const message = t(`STATUS$${agentState.toUpperCase()}`);
```

## Frontend: Data Fetching Architecture

UI components must never call API client methods (`frontend/src/api/`) directly. All data access must go through TanStack Query hooks:

```
UI components → TanStack Query hooks (frontend/src/hooks/query/ or mutation/) → API client (frontend/src/api/) → API endpoints
```

Flag any component that imports directly from `#/api/` and calls fetch/mutation functions without a TanStack Query wrapper.
