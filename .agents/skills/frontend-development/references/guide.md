## Arbitrary-value guard adoption

`shadcn/no-arbitrary-values` warns globally. Twenty-five migrated owners reject new
arbitrary classes: command-menu, backend-version, settings-nav-header,
backend-synced-settings-badge, modal-backdrop, environment-switch-overlay,
context-window-meter and conversation-panel-wrapper. Completed message/motion
owners also reject new arbitrary values: custom-toast-handlers,
form-control-classes, chat-status-indicator, context-window-ring,
user-message-body and conversation-main. Their exact paths are in `eslint.config.js`,
with executable probes in `__tests__/scripts/arbitrary-values-lint.test.ts`.
Completed Markdown code/path chips, horizontal fade consumers, file-drop artwork, automation view/filter/run-phase bounds, conversation-card reserves and the hover-reveal helper also reject new arbitrary values.
Other components, routes and primitive implementations retain warnings during
migration. Keep legitimate measured/portal geometry with its owner; do not
expand this file list or add exemptions without auditing the remaining values.

## Raw-color class-map scanning

`shadcn/no-raw-colors` rejects raw Tailwind palette utilities, undeclared named
color utilities and raw SVG paint throughout source, including primitive
implementations. The seven exact legacy JS-config/plugin color exceptions stay
valid. Backend-status-dot, Markdown blockquote and mcp-server-health additionally
use `scanAllStrings` for literal class recipes hidden behind map/helper lookups;
keep broad scanning away from unrelated program tags. Executable probes use the
real flat config in `__tests__/scripts/raw-colors-lint.test.ts`.

Bracketed hex/rgba color classes remain covered by the independently staged
`shadcn/no-arbitrary-values` policy above. Do not claim that the palette guard
finishes that migration or checks opaque runtime expressions.

## Horizontal scroll fades

`from-scroll-fade` resolves the existing `--oh-scroll-fade-from` ancestor override with `--oh-color-base` as fallback. The `scroll-fade-surface` utility sets that override to `--oh-surface` for Markdown artifact cards. Keep left/right edge visibility, direction, animation and scroll handling with their current owners; do not replace this override with a fixed palette color.

## Inline-style rollout

`shadcn/no-inline-styles` warns globally, including UI primitive implementations.
Four clean appearance owners reject inline styles at error severity: divider,
toggle-switch, combobox-caret and backend-status-dot. The exact paths live in
`eslint.config.js`, with executable probes in
`__tests__/scripts/inline-styles-lint.test.ts`; other owners retain warnings.
Numeric CSS custom properties and semantic token references still pass in both
scopes. Hardcoded variable colors and style elements remain findings.
Do not add a blanket property allowance or blindly replace measured dimensions,
portal coordinates, or renderer-library styles with static classes. Review the
value's owner before moving it to classes or CSS variables.

`src/components/providers/agent-server-ui-root.tsx` is the exact exception: its
public `style` / `styleOverrides` API forwards caller-owned styles, so the
implementation must accept an opaque object. Other lint policies keep their scope.

## Skill Loading

Public skills are loaded from the `@openhands/extensions` npm package at build time through `SKILLS_CATALOG` from `@openhands/extensions/skills`. `SkillsService` maps catalog entries to `SkillInfo` and merges them with user and project skills fetched from the Agent Server using `load_public: false`.

Bundled catalog skills use the persisted `enabled_skills` allow-list defaulted from `DEFAULT_ENABLED_SKILL_NAMES`. User and project skills remain enabled unless listed in `disabled_skills`. Keep this logic centralized in `src/utils/skill-enablement.ts`. The Agent Server no longer clones the extensions repository or uses `EXTENSIONS_REF` for public skills.


## Shared motion recipes

`src/utils/form-control-classes.ts` uses named shell and transform recipes.
Keep each transition's property list, timing fallback, duration fallback and
important modifier intact; `transition-transform` includes more properties than
the existing transform-only contract. Register custom transition and animation
names in `cn` so later caller utilities retain Tailwind's conflict precedence.

## Exact rem layout aliases

Automation popover widths, run-phase limits and conversation hover-action reserves use the existing spacing scale (`min-w-40`, `min-w-64`, `max-w-112`, `max-w-48`, `min-w-15`) instead of equivalent arbitrary rem literals. Preserve each responsive hover/focus variant. Geometry is equivalent on the default spacing scale; explicit host scale overrides follow normal named-token behavior. Do not apply this substitution to fixed pixel geometry: a rem alias would change behavior when the embedding root font size differs.

## Workspace files

Local Files discovery preferences belong to `misc_settings.app_preferences.workspace_file_discovery`, keyed by working directory on the active backend. `useWorkspaceFileDiscovery` reads the settings cache and saves sparse workspace patches. Preserve default exclusions and the 2,000-file cap when no configuration exists. Fetch one extra result to distinguish truncation from an exact-size list. Cloud keeps its server-side listing contract without these controls.

## Caret appearance contracts

`shadcn/no-restyle` recognizes direct, aliased, and namespaced imports from
`#/ui/combobox-caret`. Caret callers retain layout, icon sizing, and text-color
utilities; padding, background, shape, and effects belong to the primitive.
Use `isOpen` for rotation on `ComboboxCaretButton` and `ComboboxCaretInline`.
`ComboboxCaretIcon` is the generic SVG and still permits caller orientation.
Primitive implementations remain exempt from no-restyle. The existing narrow
static-class exceptions do not imply that opaque values are checked by no-restyle.

## Markdown code spacing

Inline code and workspace-path chips share `px-code-inline-x` / `py-code-inline-y`; plain code blocks use `p-code-block`. Their theme spacing entries retain em units so padding follows the element font. Register these spacing names in `cn()` so general and directional caller overrides keep normal precedence. The scoped library CSS and class-composition tests cover this contract.

## Message wrapping recipes

`message-word-break` retains `word-break: break-word` for chat bodies, corrective
nudges and toast content. Toasts also use Tailwind's `wrap-anywhere` for their
independent overflow-wrap behavior. Do not replace the former with `break-words`: it
sets a different CSS property. Toast-library container styles remain library-owned.

## File-drop illustration geometry

The file-drop illustration uses `file-drop-image` and `file-drop-arrow` utilities for fixed SVG pixel offsets and rotation. Preserve pixel units in embedded roots; the shipped scoped library CSS contract covers these coordinates.

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


- Vite dev mode can black-screen on first load with `504 Outdated Optimize Dep` if core client-entry deps are not prebundled; keep `react`, `react/jsx-runtime`, `react-dom/client`, and `react-router/dom` in `optimizeDeps.include`.
- Vercel deployment note: React Router builds for this repo must keep `build/client` intact on actual Vercel builds and include `presets: [vercelPreset()]` from `@vercel/react-router/vite`; flattening `build/client` during a Vercel build produces deployments with empty outputs (`routes: null`, no static files) and a production 404.

- The repo should include a root `LICENSE` file to satisfy the incubator-program requirements.
- OpenHands repo bootstrap files live under `.openhands/`:
  - `.openhands/setup.sh` installs `uv` (via `curl -LsSf https://astral.sh/uv/install.sh | sh`) if not present, installs frontend dependencies with `npm ci` when needed, creates `.env` from `.env.sample` if missing, appends `VITE_WORKING_DIR` for this repo when unset, and generates `src/i18n/declaration.ts` via `npm run make-i18n`.
- The repo now includes `.agents/skills/custom-codereview-guide.md`, adapted from `OpenHands/software-agent-sdk`, to force PR reviews to always leave either an APPROVE or COMMENT review instead of silently finishing with no review object.

- HeroUI rollback / migration notes:
  - The attempted HeroUI v3 upgrade changed global theme wiring and homepage design tokens enough that the repo currently prefers `@heroui/react@2.8.10` until a broader visual validation pass is done.
  - Keep the v2 Tailwind integration active via `@plugin '../hero.ts'` in `src/tailwind.css` and source HeroUI classes from `node_modules/@heroui/theme/dist/**/*`.
  - The settings UI currently relies on the v2 `Autocomplete` + `AutocompleteItem`/`AutocompleteSection` APIs in `settings-dropdown-input.tsx` and `model-selector.tsx`; a future v3 retry will need to replace those controls again.
- Library i18n is now namespace-scoped under `openhands`: `src/i18n/index.ts` exports `OPENHANDS_I18N_NAMESPACE`, `translationResources`, and `waitForI18n()`, `scripts/make-i18n-translations.cjs` emits `public/locales/<lang>/openhands.json`, standalone `src/entry.client.tsx` explicitly awaits i18n init, and host apps can register bundles via the `@openhands/agent-canvas/i18n` subpath export.

- Route decoupling note: `src/components/` should stay free of direct `react-router` imports. Route state now flows through `src/context/navigation-context.tsx`, the standalone app bridges router state with `src/routes/react-router-navigation-provider.tsx`, and link-like UI should use `src/components/shared/navigation-link.tsx`.
- Test helper note: `test-utils.tsx` now wraps renders with a default `NavigationProvider` (`currentPath: "/"`, `conversationId: "test-conversation-id"`). Navigation-sensitive tests can override that via `renderWithProviders(..., { navigation: { ... } })`.
- CSS isolation for embeddable/hosted use now relies on a scoped wrapper attribute: all bundled CSS is prefixed under `[data-agent-server-ui]` via `postcss-prefix-selector` in `vite.config.ts`, with selector exceptions handled by `transformAgentServerUISelector()` in `src/styles/agent-server-ui-style-scope.ts`. That transform must remap global selectors like `:root`, `html`, and `body` directly onto the scoped shell instead of emitting impossible descendants such as `[data-agent-server-ui] :root`.
- The library ships its CSS as `dist/styles.css`, exported as `@openhands/agent-canvas/styles.css`: `src/index.ts` imports the single entry `src/styles.css` (which `@import`s `index.css`, then `tailwind.css`), and `build.lib.cssFileName` names the extracted file. Keep it one CSS import: Vite's CJS output turns adjacent stripped CSS requires into a dangling comma. Hosts import it explicitly; the JS entries stay free of CSS imports. Do not add a `sideEffects` glob list to package.json: any list marks the unlisted JS side-effect free, and the library build then drops modules such as `dist/components/browser/index.js`. `__tests__/vite-config.test.ts` builds the library and checks that the stylesheet is scoped and `dist/index.cjs` parses; `__tests__/package-library.test.ts` checks the export.
- Public embedding entry points should use `AgentServerUIProviders` (scoped root on by default) or `AgentServerUIRoot` for manual control. The standalone app already renders its own scoped root in `src/root.tsx`, so `src/entry.client.tsx` must pass `withStyleRoot={false}` to avoid nesting duplicate shells. Keep `AgentServerUIRoot` and the scoping constants re-exported from `src/lib/index.ts` so library consumers can customize the host wrapper without reaching into private paths.
- `AgentServerUIRoot`'s themed inner wrapper must set a default `color: var(--foreground)` in addition to the `dark` / `data-theme` markers; otherwise inherited text and `currentColor` SVG icons fall back to dark browser defaults after CSS scoping, causing dark-on-dark regressions on pages like the home screen.
- Theme/customization tokens for the embedded shell are exposed as `--oh-*` CSS variables. Override them through `styleOverrides`, `style`, or host CSS targeting `[data-agent-server-ui]`; Tailwind theme tokens in `src/tailwind.css` should continue to reference those variables with `@theme inline` so host apps can restyle the UI without reworking component class names.
- Named control radii (`rounded-pill`, `rounded-composer`, `rounded-file-action`, `rounded-facet-control`) retain the existing absolute pixel values in `src/tailwind.css`. `cn()` registers these names in tailwind-merge's radius namespace so caller and directional overrides retain normal utility precedence. Keep the theme entries and merger registration aligned; do not replace these values with rem or percentage rounding.

- Small form feedback uses the `feedback-error` / `feedback-success` text roles (`--oh-feedback-error` / `--oh-feedback-success`), separately from the general danger/success control palette. Their built-in values must meet 4.5:1 on modal surfaces; host overrides remain owned by the embedding root. See `src/themes/README.md` and the theme contrast tests when migrating raw feedback colors.
- Regression coverage for the CSS isolation work lives in `__tests__/agent-server-ui-providers.test.tsx`, `__tests__/agent-server-ui-style-scope.test.ts`, and the browser-level CSS-isolation test in `tests/e2e/mock-llm/regressions/mock-llm-ui-regressions.spec.ts`.
- MCP credential selection: `SaveAsSecretToggle` keeps its native checkbox as the semantic control and uses `ToggleSwitchVisual` for the decorative track. Selection uses contrast/surface tokens. Keep both states' labels readable, allow long secret names to wrap with their help control, and preserve visible keyboard focus.

- Conversation history is loaded lazily, REST-first then WebSocket:
  - `useConversationHistory` (in `src/hooks/query/use-conversation-history.ts`) fetches only the most recent `INITIAL_HISTORY_PAGE_SIZE` (default 50) events using `sort_order='TIMESTAMP_DESC'`, then reverses to chronological order. Older pages are paginated in via `useLoadOlderEvents` when the user scrolls near the top of the chat.
  - `EventService.searchEvents(conversationId, conversationUrl, sessionApiKey, options)` returns the raw `EventSearchPage` (`{ items, next_page_id }`); options support `limit`, `pageId`, `sortOrder`, `timestampGte`, `timestampLt`. Both the local and cloud-proxy code paths forward the new params.
  - The main `ConversationWebSocketProvider` waits for the REST query to settle before opening its socket, then connects with `resend_mode='since'` and `after_timestamp=<latest preloaded event ts>` (falling back to `'all'` when the REST result is empty or errored). The legacy `resend_all=true` flag is removed for the main connection; the planning-agent sub-conversation still uses `resend_all` until it is migrated to the same REST-then-WS pattern.
  - The event store gained a bulk `addEvents(events)` action (used for the initial REST seed and for "scroll-up" pagination) that re-sorts by timestamp once at the end so older pages can be merged in cheaply. Per-event dedup still works via the existing `eventIds` set.
  - `ChatInterface` wires `useLoadOlderEvents` into its scroll handler (threshold 80px from the top), shows a `data-testid="loading-older-events"` spinner during pagination, and preserves the visible scroll offset by storing the previous `scrollHeight` and adding the height delta after the older page renders.

  - `useLoadOlderEvents` intentionally distinguishes between "no anchor yet" (empty store before the initial REST seed, so `loadOlder()` should no-op) and "malformed oldest event" (store has an oldest event with no timestamp, so the hook throws, flips `hasMore` false, and `ChatInterface` surfaces the failure via the shared error banner instead of failing silently).

- Activity grouping in the chat stream:
  - `groupEvents` combines consecutive tools, successful/blocked hooks and reasoning; visible agent messages/thoughts, errors, approvals and artifact previews remain boundaries. `Messages` uses a one-step minimum so a completed group never unfolds just because the agent stopped. A lone blocked hook opens directly to its details; in mixed groups its blocked badge stays on the hook row, not the summary. Failed hooks remain standalone.
  - `useSessionActivityStore` owns observed initialization/resume steps and disclosure state, scoped by backend, organization and conversation. Launch/task aliases transfer the same entry to the real conversation. This is tab-local presentation history in sessionStorage, not an SDK event or cross-device transcript; old conversations do not gain invented initialization rows.
  - A READY sandbox leaves its session in `connecting` until the conversation socket is open and pending sends have settled. `ChatInterface` completes that handoff; startup/resume owns the progress row throughout. Pending user bubbles render before the active session group, keep failure/retry/stop controls, and omit their separate sending status only during that handoff. Later ordinary sends and connection recovery retain their own feedback.
  - Session entries and streamed/finalized message reasoning join the first following activity run. Session setup is excluded from tool counts. The live header shows activity; completion keeps the same group and summarizes actual tools, or the completed session when no tools ran.
  - Count summaries use natural English singulars and comma-separated sentence fragments. Later fragments use the i18n `continuation` context; casing is translated, never applied to tool names or paths. The separator is localized.
  - A one-tool group uses the existing descriptive tool title. It reveals the body directly only when there is no session entry, reasoning or other activity alongside it; mixed groups retain collapsed tool rows until the user opens them. Its shared step disclosure records revealed output so adding another tool or moving a failed ACP call outside the group preserves that output. Reasoning, session history, rejection/timeout indicators and actionable boundaries remain accessible.
  - Startup headers show the latest reported phase, independently of deduplicated history. Setup-only groups expose observed phase details directly; groups with tools retain a nested session entry. Phase labels describe observed activity, not inferred success or skipped steps. Other sandbox/composer status consumers keep their own labels.
  - Active startup/resume groups stay visible while the initial history page is unknown. Paginated history hides completed initialization until the beginning is loaded; resume entries use an observed event anchor when available and timestamp ordering otherwise. Errors and pending secret/confirmation forms stay outside collapsed details.
  - Coverage lives in `group-events.test.ts`, `event-group.test.tsx`, `messages-model-messages.test.tsx`, and `session-activity-store.test.ts` under `__tests__`.

- Home page workspace UX (local backend):
  - `FolderBrowserModal`'s "Use this folder" button adds **only the currently navigated directory** as a single workspace (named by its basename). It no longer iterates `subdirs` and adds each child as a separate workspace.
  - The `WorkspaceDropdown` sticky footer now exposes both "+ Add Workspace" (opens the folder browser) and "Manage Workspaces" (opens `ManageWorkspacesModal`, which lets users remove individual workspaces via `useWorkspacesStore.removeWorkspace`). The Manage button is hidden when there are no workspaces yet.
  - The conversation-panel new-thread picker (`ConversationPanelNewThreadPicker` in `src/components/features/conversation-panel/conversation-panel-new-thread-picker.tsx`, mounted from `conversation-panel.tsx`) is a `FolderPlus` icon button in the conversation-list header, not a labelled "+ New Conversation" button, and it branches on `backendKind`. Local backends get `LocalNewConversationMenu`, whose popover is a **flat list**, not the home-screen combobox: a leading "No workspace" entry plus one entry per stored workspace, each clicking through to `useCreateConversation` immediately (no separate Launch button). It mirrors the dropdown footer actions/pattern (`+ Add Workspace`, `Manage Workspaces`) locally rather than embedding `WorkspaceDropdown` itself. Cloud backends get `CloudNewConversationMenu` instead, a searchable repository picker (`useGitRepositories` / `useSearchRepositories`) with no workspace entries and no footer actions. The sidebar rail's "+ New Chat" item is a nav link to `/conversations` (`src/components/features/sidebar/sidebar-rail-body.tsx`), not this popover.
  - `useResolvedWorkspaces()` now returns `isLoading` / `isError` for parent-directory scans; `WorkspaceSelectionForm` should surface that state (status text and disabling the empty dropdown while parent results are still loading) instead of assuming the merged list is immediately ready.
  - `ManageWorkspacesModal` should require a confirmation step before removing either a saved workspace or a workspace parent; parent removals should mention the child-workspace impact, and tests should assert both the confirmation flow and that removing the selected workspace clears the launch selection.
  - In `useWorkspacesStore`, keep `clearWorkspaces()` scoped to literal workspaces only; use explicit helpers like `clearWorkspaceParents()` / `clearAll()` for broader resets so future callers do not accidentally wipe parent registrations.

- Default LLM model — `DEFAULT_SETTINGS.llm_model` (`"openhands/glm-5.2"`, defined in `src/services/settings.ts`) is the canonical frontend default. `buildConfiguredOpenHandsAgentSettings` in `src/api/agent-server-adapter.ts` **always** sends this value explicitly when the resolved `llm.model` is absent, empty, or whitespace-only — the frontend never relies on the agent-server SDK's own default (`gpt-5.5`). If you change the default model, update `DEFAULT_SETTINGS.llm_model` in `src/services/settings.ts`.

- Custom secrets are NOT auto-attached by the agent-server. `POST /api/conversations` only persists what the client sends in `request.secrets`; the persisted secrets store (`/api/settings/secrets`) is never read at conversation start. `buildStartConversationRequestWithEncryptedSettings` enumerates `SecretsService.getSecrets()` and turns each entry into a `LookupSecret` whose `url` points back at `/api/settings/secrets/{name}` and whose `headers` carry `X-Session-API-Key` for auth.

- MCP page layout: MCP is a **top-level** nav entry at `/mcp` (rendered by `src/routes/mcp.tsx`), shown right below "Skills" in `src/components/features/sidebar/sidebar.tsx`. `src/routes/mcp-settings.tsx` re-exports the new page so the published `MCPSettings` library symbol (in `src/components/settings/index.ts`) keeps the same shape. The legacy `/settings/mcp` redirect was removed in issue #1337. Marketplace catalog data and MCP logo mappings live in the MCP-capable entries from `@openhands/extensions/integrations`; the Slack API catalog option should point at `https://github.com/zencoderai/slack-mcp-server` and use `@zencoderai/slack-mcp-server`. Deprecated marketplace entries removed upstream (for example GitLab / Google Maps / Postgres / Puppeteer / SQLite) should disappear from the marketplace grid. The Installed section still needs to render and search arbitrary non-catalog custom servers via the raw server `name` / `command` fallback in `src/utils/mcp-marketplace-utils.ts` + `InstalledServerCard`. Tavily is a regular stdio MCP entry (`tavily-mcp` + `TAVILY_API_KEY`), not a special built-in sentinel anymore. Components are colocated under `src/components/features/mcp-page/` and reuse the existing `MCPServerForm` for the "Add custom server" / edit flow.
- MCP catalog runtime patching: `getMcpMarketplaceCatalog()` in `src/utils/mcp-marketplace-utils.ts` pipes every catalog entry through patch functions before the UI sees it. Two patches exist: `patchLinearEntry` (rewrites the deprecated Linear SSE endpoint to streamable HTTP) and `patchGitHubEntry` (rewrites the `docker run` transport to the native `github-mcp-server stdio` binary, only when `getDeploymentMode() === "docker"`). The `getDeploymentMode()` helper is exported from `src/api/agent-server-adapter.ts` and reads the `mode` field from the runtime services info. The Docker image pre-installs the `github-mcp-server` Go binary at `/usr/local/bin/github-mcp-server` via a dedicated Dockerfile download stage (`github-mcp-download`). This avoids a Docker-in-Docker requirement — GitHub is the only catalog entry that uses `docker` as its stdio command; all others use `npx` or `uvx`. When adding similar patches for other entries, follow the same pattern: guard on `entry.id`, check environment conditionally, spread immutably.

- Library packaging notes:
  - Public npm entrypoints now come from `src/index.ts` → `src/lib/index.ts`, with domain barrels under `src/components/{conversation,terminal,browser,files,settings,sidebar}/index.ts`.
  - `npm run build` remains the standalone app build (`react-router build`), while `npm run build:lib` runs `vite build` in library mode plus `tsc -p tsconfig.lib.json` to emit `.d.ts` files into `dist/`.
  - The library build relies on `vite.config.ts` with `BUILD_LIB=true`, preserved modules in `dist/`, and package `exports` entries that map root/subpaths to `dist/**/*.js` plus matching declaration files.
  - Declaration emit needs `src/library-env.d.ts` and the narrowed `tsconfig.lib.json`; broad `src/**/*.tsx` declaration builds pulled in route-only files and missed `?react`/window globals.
- Bundle/dev-graph hygiene (Tier 1 cleanup landed):
  - `src/i18n/translation.json` (~1 MB) is imported only by `src/i18n/resources.ts`, which `src/i18n/index.ts` re-exports as `translationResources` for the `@openhands/agent-canvas/i18n` subpath. The re-export is a `export … from` plus `/* @__PURE__ */` annotation, so rollup drops the JSON from the app build (prod `custom-toast-handlers` chunk: 909 KB -> 74 KB; `conversation` chunk: 728 KB -> 392 KB). Do not move the JSON import back into `src/i18n/index.ts` — that immediately re-bundles all translations into every chunk that imports `i18n`.
  - The environment-switch overlay is split: lightweight store/triggers live in `components/features/backends/environment-switch-store.ts`; the React component lives in `environment-switch-overlay.tsx` (re-exports the store API for back-compat). Eagerly-mounted callers (e.g. `backend-selector.tsx`) MUST import trigger helpers from the store, not the overlay file. The overlay is `React.lazy`'d from `routes/root-layout.tsx`.
  - Conditional UI is `React.lazy`'d to keep eager graphs small. Current examples include `AlertBanner` and `CommandMenu` in `root-layout.tsx`, `SettingsModal` in the sidebar, and backend/onboarding modals in `root.tsx`. Tests that assert on these mounted nodes may need `await screen.findByTestId(...)` / `waitFor(...)` instead of synchronous `getByTestId(...)`.
  - The terminal tab (`components/features/terminal/terminal.tsx`) is `React.lazy`'d in `conversation-tab-content.tsx` alongside the other tabs, so xterm + addon-fit + xterm.css don't enter the conversation route's eager graph (they ship as a separate `terminal-*.js` chunk now).
  - `components/conversation-events/chat/` and `components/features/command-menu/` deliberately have no barrel `index.ts`. Import deep paths (`./messages`, `./event-message-components/<name>`, `./event-content-helpers/should-render-event`, `command-menu/command-menu`) so Vite dev doesn't fan out a barrel.



- ESLint config (flat, ESLint 9): the project uses `eslint.config.js` (not `.eslintrc`) and runs on `eslint@9.x`, not 10. The constraint pinning us below 10 is `eslint-plugin-react@7.37.x`, which still calls `context.getFilename()` at rule-load time — that API was removed in ESLint 10 and `@eslint/compat`'s `fixupPluginRules` does NOT shim it. Don't try to bump eslint past 9 until eslint-plugin-react ships a v10-compatible release. Import rules come from `eslint-plugin-import-x` (the maintained fork of `eslint-plugin-import`) but are registered under both `import-x/` and `import/` prefixes via `plugins: { import: importXPlugin, ... }` so existing `// eslint-disable-next-line import/...` directives keep working. `linterOptions.reportUnusedDisableDirectives` is set to `"warn"` (not "off") so stale airbnb-era disable comments still surface in lint output without failing CI. The TS-overrides block has an `ignores: ["src/hooks/query/query-keys.ts"]` so the `no-restricted-syntax` rule banning raw `["settings", ...]` query keys doesn't fire on the file that defines the helpers themselves. No `.npmrc` / `legacy-peer-deps` flag is needed — all our plugins declare ESLint 9 peer compatibility.

## Extended type scale

`text-4xs`, `text-xs-plus`, `text-sm-plus` and `text-title` retain existing
9px, 13px, 15px and 32px font sizes without setting line-height. Their names are
registered in cn's text-size theme so color classes stay independent and later
caller sizes or leading retain the same precedence as the former arbitrary
sizes. Keep the compiled-CSS and class-composition contracts when changing them.

## Fixed content columns

Static metadata rows use `metadata-grid-columns` (auto / 1fr); automation
summary rows use `automation-summary-columns` (auto / minmax(0, 1fr)); run-log
details use `sm:run-log-detail-columns` (5rem / minmax(0, 1fr), from 40rem).
These single-property utilities preserve intrinsic sizing and responsive scope.
They are static owner recipes, not caller className overrides.

- Executable plugin automations (`plugin_spec`) edit their configuration through the setup form's `pluginConfig` JSON field. The agent proposes that field through `automation_form_update`; only Save sends `config` and the trigger to the existing automation's PATCH endpoint. Preserve the plugin source, pins, and runtime fields. Gate configuration writes on the automation service's `pluginConfigUpdate` capability; older services display the configuration read-only.
