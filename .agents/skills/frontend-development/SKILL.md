---
name: frontend-development
description: This skill should be used when the user asks to "add UI copy", "add a translation", "optimize the frontend bundle", "change onboarding", "change conversation UI", "add a query key", "change MSW mocks", or modifies React components, hooks, routes, i18n, lazy loading, persisted UI state, or shared frontend behavior.
---

# Frontend Development

Apply Canvas-specific React, TypeScript, internationalization, state ownership, mock-mode, and bundle-performance conventions.

## Workflow

1. Read `references/guide.md` before making broad or shared frontend changes.
2. Classify every new string as localized UI copy, a named program identifier, or a constrained union tag.
3. Give durable state one owner and avoid mirroring stores into component state.
4. Preserve lazy-loading boundaries and avoid internal imports through library barrels.
5. Keep MSW fixtures in production-build-visible locations and verify static mock mode with a mock-enabled build.
6. Add focused behavioral coverage for the affected feature invariants.

## Reference

- **`references/guide.md`** — i18n and magic-string rules, generated files, mock mode, loading/performance budgets, lazy imports, query-key and lint constraints, plus feature-specific state and UI invariants.
