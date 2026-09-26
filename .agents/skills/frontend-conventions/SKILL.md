---
name: frontend-conventions
description: >
  String and identifier conventions for the frontend — user-facing strings go
  through i18n, non-UI identifiers live in named constants, union tags use
  string-literal types, and the allowed exceptions. Load before adding any
  new string, constant, or discriminated union.
triggers:
  - i18n
  - magic string
  - literal string
  - translation
  - no-magic-strings
  - union tag
---

## No Magic Strings

Avoid inline string literals when they represent reusable user-facing copy or shared program identifiers. The `i18next/no-literal-string` rule is set to `"error"` for configured JSX text and attributes, and targeted `no-restricted-syntax` rules enforce shared translation and query-key patterns. Do not claim broader lint enforcement than `eslint.config.js` provides.

### Rule 1 — User-facing strings go through i18n

Every visible string (button labels, headings, validation messages, `aria-label`, `title`, `alt`, toast copy, placeholders) **must** be routed through `react-i18next`'s `t()` keyed by an `I18nKey` enum member. Keys are declared once in `src/i18n/translation.json` with values for all 15 supported languages (see `src/i18n/index.ts::AvailableLanguages`), and `npm run make-i18n` regenerates `src/i18n/declaration.ts` + `public/locales/<lang>/openhands.json`. Run `npm run check-translation-completeness` when translations change; it is also part of the staged-file checks.

```tsx
// CORRECT
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

const { t } = useTranslation("openhands");
return (
  <button aria-label={t(I18nKey.CHAT$DISMISS_LABEL)}>
    {t(I18nKey.CHAT$DISMISS)}
  </button>
);

// WRONG -- ships English to every locale; flagged by i18next/no-literal-string
return <button aria-label="Dismiss">Dismiss</button>;
```

Key naming follows the existing `CATEGORY$IDENTIFIER` convention (see `src/i18n/translation.json` — common prefixes: `CHAT_INTERFACE$`, `SETTINGS$`, `COMMON$`, `BUTTON$`, `HOME$`, `MICROAGENT$`, etc.). Reuse an existing prefix; only introduce a new one when no sensible bucket exists.

The configured `jsx-attributes.include` list catches literal values for common user-facing attributes such as `aria-label`, `placeholder`, `title`, and `alt`. Continue to route user-facing strings through `t()` even when an uncommon prop is outside that list.

### Rule 2 — Non-UI identifiers live in named constants, not inline literals

For strings the user never sees but the program reads (storage keys, event names, query keys, route paths, env-var names, header names, hardcoded paths, feature-flag identifiers), declare a single named constant in the closest module that owns the concept and import it everywhere else. Co-locate related constants in a tiny dedicated file (`*-keys.ts`, `*-constants.ts`) when more than two callers need them.

```ts
// CORRECT
const ONBOARDING_COMPLETED_KEY = "openhands-onboarded";
localStorage.setItem(ONBOARDING_COMPLETED_KEY, "true");

// CORRECT -- query keys go through SETTINGS_QUERY_KEYS / SECRETS_QUERY_KEYS / …
//            in src/hooks/query/query-keys.ts (enforced by no-restricted-syntax)
queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });

// WRONG -- duplicated literal across files, no compile-time link, silent typo risk
localStorage.setItem("openhands-onboarded", "true");
queryClient.invalidateQueries({ queryKey: ["settings"] });
```

Already-named constants in this repo include `DEFAULT_WORKING_DIR` (`src/api/agent-server-config.ts`), `OPENHANDS_I18N_NAMESPACE` (`src/i18n/index.ts`), `BUNDLED_BACKEND_ID` (backend registry), and the `*_QUERY_KEYS` helpers in `src/hooks/query/query-keys.ts`. Reuse these instead of re-inlining the literal.

### Rule 3 — Discriminated-union tags use string-literal types, not bare strings

When a string is part of a discriminated union or enum-like set (event kinds, backend kinds, tab IDs, agent statuses, observation result statuses), the type itself should constrain the literal. Pass values typed against that union, not raw `string`, so callers get autocomplete and the compiler catches typos.

```ts
// CORRECT
type BackendKind = "local" | "cloud";
if (backend.kind === "cloud") { … }

// WRONG -- `backend.kind` typed as `string`; "clould" compiles fine
if (backend.kind === "clould") { … }
```

### Allowed exceptions

- Test fixtures (`__tests__/`, `tests/e2e/`) may use inline literals for setup data — tests are the boundary where strings stop being magic.
- Non-localizable display glyphs (keyboard shortcuts like `⌘↩`, currency symbols, etc.) may stay inline behind an `eslint-disable-next-line i18next/no-literal-string` comment. Keep the disable on the single offending line; never widen it to a file-level disable for a single glyph.
- Generated files (`src/i18n/declaration.ts`, `public/locales/<lang>/openhands.json`) are produced by `npm run make-i18n`; do not hand-edit, do not lint-target.

When adding code that needs a new string, decide up front which rule it falls under: if a user reads it → Rule 1; if the program reads it → Rule 2; if it tags a union → Rule 3. Do not commit code that fails any of these rules just because the linter happens not to catch it.
