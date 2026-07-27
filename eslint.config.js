// Flat ESLint config (ESLint 9+).
//
// Migration notes (see commit history for full context):
// - Replaces the legacy `.eslintrc` file, which ESLint 9 deprecated and
//   ESLint 10 will remove entirely.
// - Drops `eslint-config-airbnb` / `eslint-config-airbnb-typescript`: they
//   pin ESLint to v7/v8 and `@typescript-eslint/*` to v7, so they block any
//   future ESLint major bumps. The opinionated airbnb rules the codebase
//   actually leaned on (no-param-reassign, import/extensions, etc.) are
//   re-declared explicitly below, and the long list of airbnb rules the old
//   config already turned off has just been deleted instead of re-disabled.
// - Replaces `eslint-plugin-import` (only declares an `^8 || ^9` peer) with
//   the actively-maintained, flat-config-native fork `eslint-plugin-import-x`.
//   The rule names are kept under the `import/` prefix below to minimise
//   churn on existing `// eslint-disable-next-line import/...` comments.
// - `eslint-plugin-i18next` still ships only legacy config, so it is pulled
//   in via `FlatCompat` from `@eslint/eslintrc`.
// - Targets ESLint 9, not 10. The original dependabot PR proposed eslint@10,
//   but as of this commit `eslint-plugin-react` still calls
//   `context.getFilename()` (removed in ESLint 10) at rule-load time, which
//   makes a v10 bump explode on every file. Re-evaluate once that plugin
//   ships an ESLint-10-compatible release.

import { fileURLToPath } from "node:url";
import path from "node:path";

import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import tsParser from "@typescript-eslint/parser";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import reactPlugin from "eslint-plugin-react";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import jsxA11yPlugin from "eslint-plugin-jsx-a11y";
import importXPlugin from "eslint-plugin-import-x";
import prettierPlugin from "eslint-plugin-prettier";
import prettierConfig from "eslint-config-prettier";
import unusedImportsPlugin from "eslint-plugin-unused-imports";
import tanstackQueryPlugin from "@tanstack/eslint-plugin-query";
import globals from "globals";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
  resolvePluginsRelativeTo: __dirname,
});

export default [
  // Files / dirs we never want to lint.
  {
    ignores: [
      "node_modules/**",
      "dist/**",
      "build/**",
      "coverage/**",
      ".react-router/**",
      "playwright-report/**",
      "playwright-report-live/**",
      "test-results/**",
      "test-results-live/**",
      "public/mockServiceWorker.js",
      "src/i18n/declaration.d.ts",
    ],
  },

  // Base JS recommended rules.
  js.configs.recommended,

  // `eslint-plugin-import-x` ships flat configs directly. The rules are
  // registered under both `import-x/*` and `import/*` aliases so existing
  // `// eslint-disable-next-line import/*` comments still match.
  importXPlugin.flatConfigs.recommended,
  importXPlugin.flatConfigs.typescript,

  // `eslint-plugin-i18next` is still eslintrc-only — pull it in via compat.
  ...compat.extends("plugin:i18next/recommended"),

  // Project-wide settings + React/TS/etc. rules for source files.
  {
    files: ["**/*.{ts,tsx,js,jsx,mjs,cjs}"],
    // Surface (but don't fail on) `// eslint-disable-next-line ...` directives
    // that target rules this config no longer enables. The codebase has ~50
    // such directives left over from the airbnb era; "warn" lets us clean
    // them up incrementally instead of either failing CI on day one or
    // letting new stale directives accumulate silently.
    linterOptions: {
      reportUnusedDisableDirectives: "warn",
    },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parser: tsParser,
      parserOptions: {
        project: "./tsconfig.json",
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.es2024,
      },
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
      "jsx-a11y": jsxA11yPlugin,
      "@tanstack/query": tanstackQueryPlugin,
      "unused-imports": unusedImportsPlugin,
      // Alias `import-x` rules under `import/*` so existing
      // `// eslint-disable-next-line import/foo` comments keep working.
      import: importXPlugin,
      prettier: prettierPlugin,
    },
    settings: {
      react: { version: "detect" },
      // `eslint-import-resolver-typescript` resolves both TS path aliases
      // (via tsconfig) and regular node-style imports, so we don't also
      // configure the legacy `node` resolver shortcut here.
      "import-x/resolver": {
        typescript: true,
      },
    },
    rules: {
      // Recommended rule packs we want everywhere.
      ...tsPlugin.configs["eslint-recommended"].overrides[0].rules,
      ...tsPlugin.configs.recommended.rules,
      ...reactPlugin.configs.recommended.rules,
      ...reactPlugin.configs["jsx-runtime"].rules,
      ...reactHooksPlugin.configs.recommended.rules,
      ...jsxA11yPlugin.configs.recommended.rules,
      ...tanstackQueryPlugin.configs.recommended.rules,

      // Prettier integration. `eslint-config-prettier` turns off conflicting
      // stylistic rules; `eslint-plugin-prettier` re-reports prettier diffs
      // as lint errors so `npm run lint` fails on unformatted code (matching
      // the pre-flat-config behaviour).
      ...prettierConfig.rules,
      "prettier/prettier": "error",

      // Project conventions previously enforced via airbnb / custom rules.
      // Lint JSX *attributes* (not just text between tags) for hard-coded
      // user-facing strings. The plugin default (`mode: 'jsx-text-only'`)
      // never checks attribute values, which let untranslated strings like
      // `aria-label="Close"` / `placeholder="..."` slip past lint (cf. #1306).
      //
      // `jsx-only` checks every literal inside a JSX subtree, so we scope it:
      //  - jsx-attributes.include: only attributes that carry translatable
      //    text. Everything else (testId, name, color, to, href, className,
      //    data-*, …) is ignored automatically — no brittle deny-list.
      //  - callees/object-properties: re-list the plugin defaults (the option
      //    merge is shallow, so providing a key replaces it) and add
      //    `cn`/`className` so Tailwind class strings built via `cn(...)` or
      //    `{ className: "..." }` aren't flagged.
      "i18next/no-literal-string": [
        "error",
        {
          mode: "jsx-only",
          "jsx-attributes": {
            include: [
              "placeholder",
              "alt",
              "aria-label",
              "title",
              "label",
              "heading",
              "text",
            ],
          },
          callees: {
            exclude: [
              "i18n(ext)?",
              "t",
              "require",
              "addEventListener",
              "removeEventListener",
              "postMessage",
              "getElementById",
              "dispatch",
              "commit",
              "includes",
              "indexOf",
              "endsWith",
              "startsWith",
              "cn",
            ],
          },
          "object-properties": {
            exclude: ["[A-Z_-]+", "className"],
          },
        },
      ],
      "unused-imports/no-unused-imports": "error",
      "@typescript-eslint/prefer-optional-chain": "error",
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "@openhands/typescript-client/client/http-client",
              message:
                "Use typed @openhands/typescript-client clients instead of constructing HttpClient directly.",
            },
          ],
        },
      ],

      // Allow `interface Foo extends Bar<"foo"> {}` — the codebase uses this
      // discriminated-union pattern in `src/types/agent-server/**` and the
      // empty body is intentional. We still disallow `interface Foo {}` and
      // `type Foo = {}` (the default behaviour for the other two options).
      "@typescript-eslint/no-empty-object-type": [
        "error",
        { allowInterfaces: "with-single-extends" },
      ],

      // tsPlugin's `recommended` enables no-unused-vars; we already report
      // unused imports above, so make sure the TS rule still catches unused
      // locals/args (with the standard airbnb-style `_`-prefix escape hatch).
      // `caughtErrors: 'none'` keeps the v7 behaviour of allowing
      // `catch (error)` clauses where the error is intentionally unused.
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "after-used",
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          ignoreRestSiblings: true,
          caughtErrors: "none",
        },
      ],

      // React Compiler rules (added in eslint-plugin-react-hooks v5+) are
      // opt-in for projects that have actually adopted the compiler. Keep
      // them off so existing code isn't retroactively flagged.
      "react-hooks/immutability": "off",
      "react-hooks/preserve-manual-memoization": "off",
      "react-hooks/refs": "off",
      "react-hooks/set-state-in-effect": "off",

      // Match the legacy config: don't require .ts/.tsx import extensions,
      // but enforce them for everything else. See
      // https://stackoverflow.com/q/59265981 for background.
      "import-x/extensions": [
        "error",
        "ignorePackages",
        { "": "never", ts: "never", tsx: "never", js: "never", jsx: "never" },
      ],
      // The old config inherited a bunch of import rules from airbnb that
      // either don't apply or are now noise; explicitly turn off the ones
      // most likely to fire on this codebase.
      "import-x/prefer-default-export": "off",
      "import-x/no-extraneous-dependencies": "off",
      // `import-x/no-unresolved` is redundant with the TypeScript compiler:
      // `tsc` (run as `npm run typecheck` before `eslint`) already fails on
      // unresolved imports with much better error messages, and the rule
      // has known false positives with `paths`/exports-map resolution even
      // when the typescript resolver is configured. Keeping it on duplicates
      // tsc errors and produces noise on Vite's `?url` / `?worker` import
      // suffixes that ESLint can't see through.
      "import-x/no-unresolved": "off",
      // These two fire a lot of false positives on TypeScript projects that
      // import a namespace and then call methods off it (`import api from
      // './foo'; api.bar()` etc.). The TS compiler already catches anything
      // truly wrong here.
      "import-x/no-named-as-default": "off",
      "import-x/no-named-as-default-member": "off",
    },
  },

  // TypeScript-only overrides (re-applied airbnb-ish relaxations from the
  // previous `.eslintrc` overrides block).
  {
    files: ["**/*.{ts,tsx}"],
    ignores: ["src/hooks/query/query-keys.ts"],
    rules: {
      // Allow state mutation in reduce and Redux-style reducers.
      "no-param-reassign": [
        "error",
        {
          props: true,
          ignorePropertyModificationsFor: ["acc", "state"],
        },
      ],
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "Property[key.name='queryKey'] > ArrayExpression[elements.0.value='settings']",
          message:
            "Use SETTINGS_QUERY_KEYS helpers instead of raw settings query key arrays.",
        },
        {
          selector:
            "CallExpression[callee.name='t'] > Literal:first-child[value=/^[A-Z0-9_]+\\$/]",
          message: "Use I18nKey instead of raw translation key strings.",
        },
        {
          selector:
            "CallExpression[callee.property.name='t'] > Literal:first-child[value=/^[A-Z0-9_]+\\$/]",
          message: "Use I18nKey instead of raw translation key strings.",
        },
      ],
      "react/require-default-props": "off",
      "no-underscore-dangle": "off",
      "jsx-a11y/no-static-element-interactions": "off",
      "jsx-a11y/click-events-have-key-events": "off",
      // For https://github.com/airbnb/javascript/issues/1885
      "jsx-a11y/label-has-associated-control": [
        2,
        {
          required: {
            some: ["nesting", "id"],
          },
        },
      ],
      "react/prop-types": "off",
      "react/no-array-index-key": "off",
      "react-hooks/exhaustive-deps": "off",
      "react/react-in-jsx-scope": "off",
    },
  },
];
