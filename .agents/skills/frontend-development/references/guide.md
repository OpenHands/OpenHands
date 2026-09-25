## Skill Loading

Public skills are loaded from the `@openhands/extensions` npm package at build time through `SKILLS_CATALOG` from `@openhands/extensions/skills`. `SkillsService` maps catalog entries to `SkillInfo` and merges them with user and project skills fetched from the Agent Server using `load_public: false`.

Bundled catalog skills use the persisted `enabled_skills` allow-list defaulted from `DEFAULT_ENABLED_SKILL_NAMES`. User and project skills remain enabled unless listed in `disabled_skills`. Keep this logic centralized in `src/utils/skill-enablement.ts`. The Agent Server no longer clones the extensions repository or uses `EXTENSIONS_REF` for public skills.


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
- Public embedding entry points should use `AgentServerUIProviders` (scoped root on by default) or `AgentServerUIRoot` for manual control. The standalone app already renders its own scoped root in `src/root.tsx`, so `src/entry.client.tsx` must pass `withStyleRoot={false}` to avoid nesting duplicate shells. Keep `AgentServerUIRoot` and the scoping constants re-exported from `src/lib/index.ts` so library consumers can customize the host wrapper without reaching into private paths.
- `AgentServerUIRoot`'s themed inner wrapper must set a default `color: var(--foreground)` in addition to the `dark` / `data-theme` markers; otherwise inherited text and `currentColor` SVG icons fall back to dark browser defaults after CSS scoping, causing dark-on-dark regressions on pages like the home screen.
- Theme/customization tokens for the embedded shell are exposed as `--oh-*` CSS variables. Override them through `styleOverrides`, `style`, or host CSS targeting `[data-agent-server-ui]`; Tailwind theme tokens in `src/tailwind.css` should continue to reference those variables with `@theme inline` so host apps can restyle the UI without reworking component class names.
- Regression coverage for the CSS isolation work lives in `__tests__/agent-server-ui-providers.test.tsx`, `__tests__/agent-server-ui-style-scope.test.ts`, and the browser-level CSS-isolation test in `tests/e2e/mock-llm/regressions/mock-llm-ui-regressions.spec.ts`.

- Conversation history is loaded lazily, REST-first then WebSocket:
  - `useConversationHistory` (in `src/hooks/query/use-conversation-history.ts`) fetches only the most recent `INITIAL_HISTORY_PAGE_SIZE` (default 50) events using `sort_order='TIMESTAMP_DESC'`, then reverses to chronological order. Older pages are paginated in via `useLoadOlderEvents` when the user scrolls near the top of the chat.
  - `EventService.searchEvents(conversationId, conversationUrl, sessionApiKey, options)` returns the raw `EventSearchPage` (`{ items, next_page_id }`); options support `limit`, `pageId`, `sortOrder`, `timestampGte`, `timestampLt`. Both the local and cloud-proxy code paths forward the new params.
  - The main `ConversationWebSocketProvider` waits for the REST query to settle before opening its socket, then connects with `resend_mode='since'` and `after_timestamp=<latest preloaded event ts>` (falling back to `'all'` when the REST result is empty or errored). The legacy `resend_all=true` flag is removed for the main connection; the planning-agent sub-conversation still uses `resend_all` until it is migrated to the same REST-then-WS pattern.
  - The event store gained a bulk `addEvents(events)` action (used for the initial REST seed and for "scroll-up" pagination) that re-sorts by timestamp once at the end so older pages can be merged in cheaply. Per-event dedup still works via the existing `eventIds` set.
  - `ChatInterface` wires `useLoadOlderEvents` into its scroll handler (threshold 80px from the top), shows a `data-testid="loading-older-events"` spinner during pagination, and preserves the visible scroll offset by storing the previous `scrollHeight` and adding the height delta after the older page renders.

  - `useLoadOlderEvents` intentionally distinguishes between "no anchor yet" (empty store before the initial REST seed, so `loadOlder()` should no-op) and "malformed oldest event" (store has an oldest event with no timestamp, so the hook throws, flips `hasMore` false, and `ChatInterface` surfaces the failure via the shared error banner instead of failing silently).

- Action grouping in the chat stream:
  - `src/components/conversation-events/chat/group-events.ts` folds runs of consecutive groupable events (regular `ActionEvent`/`ObservationEvent` cards, but not `FinishAction`, `ThinkAction`, `PlanningFileEditorObservation`, `TaskTrackerObservation`, hooks, errors, or message events) into single `RenderedItem` groups. The threshold lives in `EVENT_GROUP_MIN_SIZE` (currently 2, so even pairs of back-to-back actions get folded).
  - `EventGroup` (`src/components/conversation-events/chat/event-message-components/event-group.tsx`) is the collapsible header that wraps each run. Default state is collapsed; the header shows `EVENT_GROUP$ACTIONS_COMPLETED` (with a success check) when the group is done, or `EVENT_GROUP$ACTIONS_PROGRESS` plus the currently-running action's title (from `getEventContent`) while a member `ActionEvent` has not yet been replaced by its observation in the UI events array. Expanding renders the original `EventMessage`s verbatim so each card still expands the way it did before.
  - Agent thoughts attached to an `ActionEvent` (`event.thought`) are hoisted out of groups: `groupEvents` emits a third `RenderedItem` kind `"thought"` whenever a groupable event carries (or, for an observation, originates from) a non-empty thought, flushing the current run and starting a new one. `messages.tsx` renders that item via `ThoughtEventMessage` and passes `suppressThought` to `EventMessage` so the inline thought isn't duplicated inside the group's expanded content. `ThinkAction` is excluded from this hoisting because the thought IS its action body and is rendered through its own codepath.
  - `groupEvents` now de-duplicates hoisted thoughts by action ID so mixed UI arrays that temporarily contain both an action and its replacement observation do not emit the same thought twice; `minSize` is treated as a validated internal invariant (`>= 1`).
  - `EventGroup` should return `null` for an empty `events` array and wire the toggle button to the expanded body with `aria-controls` / `role="region"` / `aria-labelledby`.
  - `src/components/conversation-events/chat/messages.tsx` is the only consumer; the grouping is transparent to upstream code. Coverage lives in `__tests__/components/conversation-events/chat/group-events.test.ts` (pure logic, including thought hoisting) and `__tests__/components/conversation-events/chat/event-message-components/event-group.test.tsx` (rendering/interaction).

- Home page workspace UX (local backend):
  - `FolderBrowserModal`'s "Use this folder" button adds **only the currently navigated directory** as a single workspace (named by its basename). It no longer iterates `subdirs` and adds each child as a separate workspace.
  - The `WorkspaceDropdown` sticky footer now exposes both "+ Add Workspace" (opens the folder browser) and "Manage Workspaces" (opens `ManageWorkspacesModal`, which lets users remove individual workspaces via `useWorkspacesStore.removeWorkspace`). The Manage button is hidden when there are no workspaces yet.
  - The conversation-panel new-thread picker (`ConversationPanelNewThreadPicker` in `src/components/features/conversation-panel/conversation-panel-new-thread-picker.tsx`, mounted from `conversation-panel.tsx`) is a `FolderPlus` icon button in the conversation-list header, not a labelled "+ New Conversation" button, and it branches on `backendKind`. Local backends get `LocalNewConversationMenu`, whose popover is a **flat list**, not the home-screen combobox: a leading "No workspace" entry plus one entry per stored workspace, each clicking through to `useCreateConversation` immediately (no separate Launch button). It mirrors the dropdown footer actions/pattern (`+ Add Workspace`, `Manage Workspaces`) locally rather than embedding `WorkspaceDropdown` itself. Cloud backends get `CloudNewConversationMenu` instead, a searchable repository picker (`useGitRepositories` / `useSearchRepositories`) with no workspace entries and no footer actions. The sidebar rail's "+ New Chat" item is a nav link to `/conversations` (`src/components/features/sidebar/sidebar-rail-body.tsx`), not this popover.
  - `useResolvedWorkspaces()` now returns `isLoading` / `isError` for parent-directory scans; `WorkspaceSelectionForm` should surface that state (status text and disabling the empty dropdown while parent results are still loading) instead of assuming the merged list is immediately ready.
  - `ManageWorkspacesModal` should require a confirmation step before removing either a saved workspace or a workspace parent; parent removals should mention the child-workspace impact, and tests should assert both the confirmation flow and that removing the selected workspace clears the launch selection.
  - In `useWorkspacesStore`, keep `clearWorkspaces()` scoped to literal workspaces only; use explicit helpers like `clearWorkspaceParents()` / `clearAll()` for broader resets so future callers do not accidentally wipe parent registrations.

- Default LLM model — `DEFAULT_SETTINGS.llm_model` (`"openhands/glm-5.2"`, defined in `src/services/settings.ts`) is the canonical frontend default. `buildConfiguredOpenHandsAgentSettings` in `src/api/agent-server-adapter.ts` **always** sends this value explicitly when the resolved `llm.model` is absent, empty, or whitespace-only — the frontend never relies on the agent-server SDK's own default (`gpt-5.5`). If you change the default model, update `DEFAULT_SETTINGS.llm_model` in `src/services/settings.ts` **and** the checklist in `specs/llm-defaults.md`. Spec: `@spec LLD-001`.

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
  - Avoid importing app code through `#/components/conversation-events/chat` or its `event-message-components/index.ts` barrel — they exist for `lib/index.ts` (npm subpath) consumers only. Internal callers use deep paths (`./messages`, `./event-message-components/<name>`, `./event-content-helpers/should-render-event`) so Vite dev doesn't fan out the barrel.



- ESLint config (flat, ESLint 9): the project uses `eslint.config.js` (not `.eslintrc`) and runs on `eslint@9.x`, not 10. The constraint pinning us below 10 is `eslint-plugin-react@7.37.x`, which still calls `context.getFilename()` at rule-load time — that API was removed in ESLint 10 and `@eslint/compat`'s `fixupPluginRules` does NOT shim it. Don't try to bump eslint past 9 until eslint-plugin-react ships a v10-compatible release. Import rules come from `eslint-plugin-import-x` (the maintained fork of `eslint-plugin-import`) but are registered under both `import-x/` and `import/` prefixes via `plugins: { import: importXPlugin, ... }` so existing `// eslint-disable-next-line import/...` directives keep working. `linterOptions.reportUnusedDisableDirectives` is set to `"warn"` (not "off") so stale airbnb-era disable comments still surface in lint output without failing CI. The TS-overrides block has an `ignores: ["src/hooks/query/query-keys.ts"]` so the `no-restricted-syntax` rule banning raw `["settings", ...]` query keys doesn't fire on the file that defines the helpers themselves. No `.npmrc` / `legacy-peer-deps` flag is needed — all our plugins declare ESLint 9 peer compatibility.
