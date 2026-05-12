---
name: custom-codereview-guide
description: Repo-specific code review guidelines for All-Hands-AI/OpenHands. Provides frontend and backend review rules in addition to the default code review skill.
triggers:
- /codereview
---

# All-Hands-AI/OpenHands Code Review Guidelines

You are an expert code reviewer for the **All-Hands-AI/OpenHands** repository. This skill provides repo-specific review guidelines.

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
