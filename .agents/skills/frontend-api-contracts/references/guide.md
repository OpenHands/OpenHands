## Frontend API Adaptation

- Frontend API adaptation lives mainly in `src/api/`:
  - `option-service` fabricates a web-client config and reads models/providers through `@openhands/typescript-client` LLM endpoints.
  - `settings-service` uses `@openhands/typescript-client` settings APIs for persistence; reads schemas from `/api/settings/agent-schema` and `/api/settings/conversation-schema`, fetches settings with optional `X-Expose-Secrets: encrypted` for conversation-start payloads, and saves settings via PATCH with diffs.
  - `agent-server-conversation-service`, `event-service`, `agent-server-git-service`, and `skills-service` route local Agent Server access through `@openhands/typescript-client` rather than direct HTTP calls.
- Current Cloud behavior is implemented explicitly through the backend registry, Cloud service layer, and device authorization flow.


## Additional Notes

- **Published binary auth fix**: When users install the npm package globally (`npm install -g @openhands/agent-canvas`) and run `agent-canvas`, the pre-built static frontend has NO `VITE_SESSION_API_KEY` baked in (npm publish runs `npm run build` with no such env var). The runtime session key is generated when the CLI launches and reaches the frontend via `scripts/static-server.mjs --session-api-key <key>`, which injects a `<head>` script that does two things: (a) sets `window.__AGENT_CANVAS_SESSION_API_KEY__ = <key>` — read by `getBakedSessionApiKey()` in `src/api/agent-server-config.ts` as a fallback when the env var is empty, symmetric with `__AGENT_CANVAS_AUTH_REQUIRED__` / `isAuthRequired()`; (b) writes the same key into `localStorage['openhands-agent-server-config'].sessionApiKey`, always overwriting when the value differs, so any code path that still reads the legacy storage key (e.g. e2e fixtures) sees the live key. The window-global path is the load-bearing one — without it, `makeDefaultLocalBackend()` returns null on a fresh install, the backend registry seeds empty, and `root.tsx` traps the user behind the Manage Backends modal instead of onboarding. `scripts/dev-with-automation.mjs` and `scripts/dev-static.mjs` both pass `--session-api-key ${config.sessionApiKey}` when starting the static server.

- Direct `dependencies` and `devDependencies` in `package.json` are exact-pinned (no caret ranges); reproducible installs should use the committed `package-lock.json` plus `npm ci`, and targeted transitive fixes still belong in `overrides`.
- Current `overrides` in `package.json` and the advisories they address (keep this list in sync when adding/removing overrides):
  - `@vercel/static-config > ajv: 8.20.0` — **GHSA-2g4f-4pwh-qvx6** (ReDoS in ajv's `$data` option, affects 7.0.0-alpha.0–8.17.1, reaches us through `@vercel/react-router` → `@vercel/static-config` → `ajv@8.6.3`). The override is intentionally **scoped** to `@vercel/static-config`: a top-level `ajv` override forces ajv 8.x everywhere and breaks ESLint's `@eslint/eslintrc`, which requires ajv ^6.x (incompatible API). Scoping lets ESLint keep its nested `ajv@6.x` while only the vulnerable consumer is bumped.
  - `dompurify: 3.4.14` — keeps Monaco's exact-pinned `dompurify@3.2.7` on a patched release for **GHSA-55q2-fjhq-7xh7**, **GHSA-39q2-94rc-95cp**, and related XSS bypasses.
  - `typed-rest-client > qs: 6.15.3` — overrides the vulnerable `qs@6.15.1` exact-pinned by every published `typed-rest-client` release in the `@stryker-mutator/core` dependency tree.
- When bumping pinned versions, use npm to update `package.json` and `package-lock.json` together; do not hand-edit only one of them.
- `npm test` now runs `npm run make-i18n` first so clean environments generate `src/i18n/declaration.ts` before Vitest loads aliased imports.
- `__tests__/vite-config.test.ts` should import `vite.config` directly under `// @vitest-environment node`; spawning plain `node -e 'import ./vite.config.ts'` is not portable across Node patch releases in CI.
- `vitest.setup.ts` must guard DOM-specific globals (`HTMLCanvasElement`, `HTMLElement`, `window`) because some suites run in the Node environment instead of jsdom.
- WebSocket hook regression note: `__tests__/hooks/use-websocket.test.ts`'s `onClose` callback assertion was flaky against the shared MSW websocket server in CI; keep that single test on a deterministic stubbed `WebSocket` close path instead of relying on MSW close timing.
- Library i18n regression note: `__tests__/i18n/library-namespace.test.ts` imports `../../src/index`, which can take >5s under the full Vitest suite after `vi.resetModules()`. Keep an explicit per-test timeout (currently 15s) so the suite doesn't fail on slow workers.

- `src/components/shared/buttons/styled-tooltip.tsx` should keep HeroUI tooltip animations disabled in Vitest (`disableAnimation` when `import.meta.env.MODE === "test"`); otherwise full-suite runs can end with unhandled `window is not defined` rejections from `framer-motion` after jsdom teardown (seen via `recent-conversation` tests in CI).
- `@openhands/typescript-client` should be pinned to a released npm version rather than an unreleased commit SHA; when agent-canvas needs new client API, release the client first and then update the dependency. Released versions should include the typed clients, agent-server version compatibility helpers, `WorkspacesClient`, `ConversationClient.switchLLM`, and subpath exports for `events/remote-events-list` and `workspace/remote-workspace` needed by the agent-canvas agent-server integration. `RemoteWorkspace.gitChanges`/`gitDiff` accept an optional `{ ref }` option; agent-canvas passes `'HEAD'` so the changes panel reflects working-tree + index versus the latest commit (i.e. staged + unstaged) instead of a diff against the upstream/default branch.
- If a GitHub-hosted git dependency is introduced, npm may normalize its lockfile URL to SSH. `scripts/vercel-install.sh` (wired through `vercel.json`) rewrites GitHub SSH URLs to HTTPS before `npm ci`; keep that generic protection in sync with any future git dependencies.

## API Access Rules

Two strict conventions govern every REST call in the frontend. Violations break CI
via `src/api/no-direct-agent-server-calls.test.ts`.

### Rule 1 -- Agent-server calls must use `@openhands/typescript-client`

All calls that target the local agent-server (`/api/*`, `/server_info`, `/sockets`)
**must** go through typed client classes from `@openhands/typescript-client`, **never**
raw `axios`, `fetch`, or the legacy shared `openHands` axios instance.

Available clients and their subpath imports:

- `ConversationClient` -- `@openhands/typescript-client/clients`
- `FileClient` -- `@openhands/typescript-client/clients`
- `VSCodeClient` -- `@openhands/typescript-client/clients`
- `ServerClient` -- `@openhands/typescript-client/clients`
- `RemoteWorkspace` -- `@openhands/typescript-client/workspace/remote-workspace`
- `RemoteEventsList` -- `@openhands/typescript-client/events/remote-events-list`

Client options are always assembled via helpers in `src/api/agent-server-client-options.ts`:

- `getAgentServerClientOptions(overrides?)` -- for SDK client constructors
- `getAgentServerHttpClientOptions(overrides?)` -- for typed wrappers such as `RemoteEventsList`; application code must not import or construct the low-level `HttpClient`

These helpers read host, session API key, and working directory from the active backend
registry and env config, so callers never hardcode URLs or auth tokens.

```ts
// CORRECT
const data = await new ConversationClient(
  getAgentServerClientOptions(),
).getConversation(id);
const file = await new FileClient(
  getAgentServerClientOptions(),
).downloadTextFile(path);

// WRONG -- raw axios/fetch calls fail the no-direct-agent-server-calls.test.ts guard
const data = await axios.get(`${host}/api/conversations/${id}`);
const data = await fetch(`/api/conversations/${id}`);
```

**Allowed exceptions** (files that may use axios directly for infrastructure reasons):

- `src/api/automation-service/automation-service.api.ts`
- `src/api/cloud/proxy.ts` -- the proxy envelope POST itself
- `src/api/main-app-auth.ts` -- the local main-app authentication endpoint

### Rule 2 -- Cloud App-API routes must go through `callCloudProxy`

Calls from the browser to the **cloud App API** (`app.all-hands.dev`) -- the
endpoints that live on the cloud backend host, not a per-conversation runtime
sandbox -- **must** go through `callCloudProxy()` in `src/api/cloud/proxy.ts`.
The SaaS now permits CORS for API-key-authenticated browser requests
(`ApiKeyAwareCORSMiddleware`), so `callCloudProxy` issues these calls **directly**
from the browser to `backend.host`; it no longer tunnels them through a
server-side proxy.

```ts
import { callCloudProxy } from "../cloud/proxy";

// CORRECT -- cloud App-API endpoint, called directly with bearer auth
const result = await callCloudProxy<ResponseType>({
  backend,
  method: "GET",
  path: `/api/v1/app-conversations/search?${params}`,
});

// WRONG -- direct axios/fetch to a cloud host bypasses the typed cloud transport
const result = await axios.get(`${backend.host}/api/v1/app-conversations`);
```

**Runtime-sandbox calls are NOT proxied.** Per-conversation runtime hosts
(`*.prod-runtime.all-hands.dev`) are reached **directly** with the typed client
(`ConversationClient` / `BashClient` / `RemoteWorkspace` / `FileClient`, or a
typed wrapper built via `getAgentServerHttpClientOptions`) pointed at the
conversation's `conversation_url`, authenticated with its session API key -- the
same path local mode uses. Do **not** route runtime calls through
`callCloudProxy` with `hostOverride`: that used to POST an envelope to
`/api/cloud-proxy` on the local agent-server, but that endpoint was **removed**
from the agent-server (software-agent-sdk PR #3326) and is absent on the
SaaS/enterprise backend, so such calls return **405**. If you find a cloud runtime
branch still using `hostOverride`, migrate it to the direct typed-client path.

`callCloudProxy` key options:

- `backend` -- the cloud `Backend` object (provides host and bearer token)
- `authMode` -- `"bearer"` (default, cloud App API) | `"none"`
  (`hostOverride` / `"session-api-key"` is legacy; the `/api/cloud-proxy` envelope
  it relied on no longer exists -- see above)

Standard cloud/local branch pattern used throughout the service layer:

```ts
if (getActiveBackend().backend.kind === "cloud") {
  return callCloudProxy({ backend: active, ... });
}
return new ConversationClient(getAgentServerClientOptions()).someMethod(...);
```

## Backend and Contract Invariants

- Use `@openhands/typescript-client` classes directly for agent-server-backed REST/workspace/event/VS Code calls. Centralize host/session API key/working-directory option assembly through `src/api/agent-server-client-options.ts`; the backend fallback policy itself lives in `src/api/backend-registry/active-store.ts`.
- Local verification/build gotchas:
  - `npm run typecheck` assumes generated translation types exist; run `npm run make-i18n` first if `src/i18n/declaration.ts` is missing.
- Original OpenHands hosted routes were removed, while current Cloud behavior is implemented explicitly through the backend registry and `src/api/cloud/`. Use `src/routes.ts` as the source of truth and do not restore old project-management, invitation, account, or git-settings routes from upstream without restoring their full API and i18n dependencies.

- `npm run dev:mock` needs MSW handlers for the direct agent-server routes used by the adapted frontend, not the original OpenHands mock paths. Key routes that must stay covered are:
  - bootstrap/model loading: `/server_info`, `/api/llm/models/verified`, `/api/llm/providers`
  - settings schemas: `/api/settings/agent-schema`, `/api/settings/conversation-schema`
  - settings CRUD: `GET /api/settings`, `PATCH /api/settings`
  - secrets CRUD: `GET /api/settings/secrets` (list), `GET /api/settings/secrets/:name` (value), `PUT /api/settings/secrets` (upsert), `DELETE /api/settings/secrets/:name`
  - conversation browsing/loading: `/api/conversations/search`, `/api/conversations?ids=...`, `/api/conversations/:id`, `/api/conversations/:id/events/*`
  - runtime git panels: `/api/git/changes`, `/api/git/diff`
- Files that MSW handlers import (demo bundles, fixture JSON) must live under `src/fixtures/` and be imported via `#/fixtures/...`. `.dockerignore` excludes `tests/` and `__tests__/` from the Docker build context, so a handler importing from those directories builds locally but fails `npm run build` inside the image (`UNRESOLVED_IMPORT`).
- Static mock verification needs a build created with `VITE_MOCK_API=true` (use `npm run build:mock`); the client must start MSW whenever that flag is enabled, even in production/static builds, otherwise routes like `/settings` and the conversations pane fall through to the static server and crash on undefined `.filter`/`.map` assumptions.
- Frontend compatibility is enforced by `assertAgentServerVersionIsSupported()` in `src/api/agent-server-compatibility.ts`, using `compatibility.minimumAgentServer` from `config/defaults.json`. `OptionService.getConfig()` calls `loadAgentServerInfo()` to enforce that floor, detect unavailable/auth-failing servers, and cache `usable_tools` for tool gating.
- Backend registry: there is no longer a separate "bundled" backend. On first read of the `openhands-backends` localStorage key (`raw === null`), `readStoredBackends()` seeds the registry with one default local backend (`makeDefaultLocalBackend()`, id `BUNDLED_BACKEND_ID = "default-local"`, host/api-key from `agent-server-config`). After that the seed is just an ordinary registered backend — users can rename or remove it like any other. `getEffectiveLocalBackend()` returns the first registered local, falling back to a synthesized default if the registry has no locals (used by API clients that need a baseline `local` target). The "Manage backends" modal and the BackendSelector dropdown both read from the single registered list, so the seeded default appears in both without any special-casing.
- Shared `Dropdown` open behavior: when the menu opens, it clears the input/search text so callers can show the current selection via `placeholder` while still rendering the full option list. Generic dropdown tests should not expect the selected label to remain in the input after reopening unless the parent explicitly controls that display.
- `useLoadOlderEvents` needs ref-based `isLoading` / `hasMore` guards in addition to React state because `ChatInterface` can trigger pagination from `onScroll`, `onWheel`, and the no-overflow effect in the same tick; closure-based state alone allows duplicate page requests.
- `ChatInterface` continuity tests should assert that conversation messages render without the full `chat-messages-skeleton`, not that `data-testid="loading-spinner"` is absent: the lazy older-events indicator reuses the shared `LoadingSpinner` component and legitimately renders that inner test id while history backfill is running.
- `useConversationHistory` now mirrors the older-events pagination fallback when the first page is exactly `INITIAL_HISTORY_PAGE_SIZE`: treat `next_page_id` **or** a full page as `hasMore`, so older agent-server variants that omit `next_page_id` still allow one more backfill request. The hook and `useLoadOlderEvents` also defensively reject mocked/malformed `page.items` responses before reversing them.

- `/server_info` tool capability metadata from `software-agent-sdk` PR #3028 ended up shipping as `usable_tools` (not `available_tools`). Frontend browser-tool gating should key off `usable_tools`, and still default to allowing tools when the server does not advertise tool metadata.

- Useful regression tests for mock mode live in `__tests__/api/option-service.test.ts`, `__tests__/api/mock-conversation-handlers.test.ts`, and `__tests__/api/mock-settings-handlers.test.ts`.
- Agent-server compatibility conventions:
  - Authenticated schema and settings calls must send the configured `X-Session-API-Key`.
  - Local provider/model discovery uses `/api/llm/providers`, `/api/llm/models`, and `/api/llm/models/verified`; Cloud model/provider search goes through the Cloud service layer.
  - `GET /api/conversations` uses repeated `ids` query parameters (`?ids=a&ids=b`).
  - Runtime git panels prefer the conversation's reported `workspace.working_dir`.
  - Conversation-start payloads use SDK-registered snake-case tool names such as `terminal`, `file_editor`, `task_tracker`, and `browser_tool_set`.
  - The `/server_info` bootstrap uses a 5-second timeout and surfaces unavailable, authentication, and unsupported-version states through the backend recovery UI.

- Git provider tokens are stored exclusively on the agent-server via `SecretsService` (`PUT /api/settings/secrets`). They are NOT mirrored to localStorage; the frontend reads which providers are connected from `settings.provider_tokens_set` (populated by `GET /api/settings`). Older notes about an `openhands-agent-server-git-provider-tokens` localStorage key are obsolete — no such key is read or written anywhere in the codebase.
- App-level user preferences (language, sound notifications, analytics consent, git identity, disabled skills) are persisted server-side under `PersistedSettings.misc_settings.app_preferences` since agent-server 1.27. `misc_settings` is a generic container for frontend-owned settings the agent doesn't interpret; `app_preferences` is currently its only nested category, but additional categories (e.g. a future `ui_preferences` for sidebar layout / view modes) drop in as additional siblings without churning the top-level wire shape. `GET /api/settings` returns the block under `misc_settings.app_preferences`, and `PATCH /api/settings` accepts a `misc_settings_diff` that is **deep-merged** into the persisted block (same semantics as `agent_settings_diff` / `conversation_settings_diff`). Partial diffs like `{"misc_settings_diff": {"app_preferences": {"language": "fr"}}}` update only the named nested field; sibling `app_preferences` fields are left untouched. Lists (`disabled_skills`) are replaced wholesale by the deep-merge. `SettingsService.transformApiResponse` hoists the nested fields onto the flat `Settings` shape so the rest of the GUI keeps reading them as top-level keys (`settings.language`, `settings.disabled_skills`, …). Do NOT reintroduce a localStorage fallback for these fields — `disabled_skills` and the rest belong in `misc_settings_diff.app_preferences`. The previous flat `app_preferences` / `app_preferences_diff` API (introduced in SDK PR #3539, never shipped to users) was replaced by the `misc_settings` container before either side reached a stable release; on-disk v2 settings files written by the flat shape are migrated automatically on first read by the agent-server. The legacy localStorage migration (`src/api/settings-service/legacy-app-preferences-migration.ts`) was removed in issue #1337 after a one-release drain period — it is no longer needed.
- Auth modes for `agent-canvas` (dev and production):
  - **Local mode** (default, no `--public` flag): A session API key is auto-generated and persisted to `~/.openhands/agent-canvas/session-api-key.txt`. The key is baked into the Vite dev server via `VITE_SESSION_API_KEY` or injected into static builds via `static-server.mjs --session-api-key`. Users never need to paste a key.
  - **Public mode** (`--public` flag): Requires `LOCAL_BACKEND_API_KEY` env var. The key is used as the agent-server session key (`OH_SESSION_API_KEYS_0`) but is NOT baked into the frontend (no `VITE_SESSION_API_KEY`, no `--session-api-key` to static-server). The frontend detects a 401 from `/server_info` via `isAgentServerAuthError()` and shows `ApiKeyEntryScreen` (`src/components/features/backends/api-key-entry-screen.tsx`). The screen reuses `BackendForm` with the host pre-filled (read-only) and prompts for the API key. On submit, the key is persisted to `localStorage['openhands-agent-server-config']` and the page reloads.
  - Dev usage: `LOCAL_BACKEND_API_KEY=my-secret npm run dev -- --public`
  - Production usage: `LOCAL_BACKEND_API_KEY=my-secret npx @openhands/agent-canvas --public`
  - The `--public` flag is supported by both `scripts/dev-with-automation.mjs` (parsed in `parseArgs()`, propagated via `config.isPublic`) and `bin/agent-canvas.mjs` (passed as `isPublic` to `main()`).
  - The 401 detection lives in `src/api/agent-server-compatibility.ts` (`isAgentServerAuthError()`), and the gate is in `src/root.tsx`'s `App` component, between the `AgentServerUnavailableError` check and the `<Outlet />` render.
  - **Key rotation resilience (non-public):** `syncLauncherDefaultLocalBackend()` in `src/api/backend-registry/storage.ts` re-runs at module init: for any stored backend whose id is `"default-local"` and whose host matches (or is loopback-equivalent to) the launcher's default, its `apiKey` is overwritten with the current `makeDefaultLocalBackend().apiKey` (sourced from `VITE_SESSION_API_KEY` or, in the published-binary path, `window.__AGENT_CANVAS_SESSION_API_KEY__`). E2E coverage: `tests/e2e/mock-llm/backends/mock-llm-auth-modes.spec.ts` (fresh-install, key-rotation, and public-mode scenarios).
- Backend/footer actions that launch modals from inside a dropdown or popover should intercept `onMouseDown` to keep the menu mounted, then perform the actual open on `onClick`. Current examples: `Add backend` / `Manage backends` in `src/components/features/backends/backend-selector.tsx`, plus the mirrored workspace-footer buttons in `src/components/features/conversation-panel/local-new-conversation-menu.tsx`.
- `BackendSelector`'s cloud-org switch paths should never rethrow from the dropdown `onChange` handler: unexpected non-Axios failures need a generic error toast instead of an unhandled promise rejection, and the malformed `(cloud backend, null org)` self-heal path should fall back to the bundled backend if `/switch` fails.
- `LocalNewConversationMenu` should support keyboard dismissal (`Escape`) for its inline popover, while still keeping the popover open when its modal children (`FolderBrowserModal`, `ManageWorkspacesModal`) are active.


- Backend dropdown connectivity indicator: `useBackendsHealth` (`src/hooks/query/use-backends-health.ts`) polls each registered backend every 10s. Local backends validate the configured session key through `SettingsClient` before calling `ServerClient.getServerInfo()` and enforcing the compatibility floor. Cloud API-key backends use `getCurrentCloudApiKey()`; cookie-auth Cloud backends use `getCloudOrganizations()`. Verdicts are surfaced as a colored dot rendered through `DropdownOption.prefix`; the trigger reads its prefix from the live `options` array (not downshift's frozen `selectedItem`) so the indicator updates without remounting. The same dot is rendered in each row of `ManageBackendsModal`, which opts into a one-shot re-probe for previously disabled backends. Tests live in `__tests__/hooks/query/use-backends-health.test.tsx`, the `connection indicator` block of `__tests__/components/backends/backend-selector.test.tsx`, and `__tests__/components/backends/manage-backends-modal.test.tsx`.

- Manage Backends modal: `src/components/features/backends/manage-backends-modal.tsx` lets users edit (host/name/api-key/kind) and remove existing backends, plus add new ones inline via a "+ Add Backend" footer button that opens a `BackendFormModal`. Both the dropdown footer's "Add backend" and the manage modal's "+ Add Backend" reuse `BackendFormModal` (see `backend-form-modal.tsx`), with `mode="add"` or `mode="edit"`; `AddBackendModal` is now a thin compatibility wrapper for `BackendFormModal mode="add"`. The modal is also auto-rendered (with a no-op `onClose`) by `src/root.tsx` when the active backend is unreachable, replacing the old full-screen `MissingAgentServerNotice` onboarding screen.

- Conversation right-panel regression note: `ConversationTabs` now owns the moved refresh/build buttons, so `__tests__/components/features/conversation/conversation-tabs.test.tsx` should cover that behavior directly. The drawer's open/closed state (`isRightPanelShown` / `hasRightPanelToggled`) is intentionally **session-only**: it always starts closed on app load (or on opening a fresh/existing conversation after a restart), but it survives in-app navigation because the Zustand `useConversationStore` stays alive across React Router transitions. The `ConversationState` localStorage blob (`conversation-state-{id}`) deliberately does **not** carry a `rightPanelShown` field — `useConversationLocalStorageState` does not expose a `setRightPanelShown` setter, `sanitizeStoredState` strips the legacy `rightPanelShown` key from older persisted blobs on read, and `RightPanelToggle` / `useSelectConversationTab` only mutate the in-memory store. In tests, seed the Zustand store directly for `selectedTab` / `isRightPanelShown` / `hasRightPanelToggled` (the component sync effect currently restores only `selectedTab` from localStorage, so localStorage alone will not make a tab read as active or the drawer read as open).

- Changes tab / `FileDiffViewer` deleted-file note: the agent-server's `/api/git/diff` endpoint calls `path.exists()` first (see `openhands-sdk/openhands/sdk/git/git_diff.py` → `get_git_diff`), so requesting a diff for a `D` (deleted) file returns `GitPathError` → HTTP 400 and trips the global QueryCache error toast. `useUnifiedGitDiff` disables the query when `type === "D"` and `FileDiffViewer` renders a localized "file deleted" placeholder (`DIFF_VIEWER$FILE_DELETED`, `data-testid="file-deleted-message"`) instead of the view-mode toolbar / Monaco editor for that case.

- Onboarding modal: `src/components/features/onboarding/onboarding-modal.tsx` is rendered by `<OnboardingHost />` and normally gated by the `openhands-onboarded` localStorage flag. `OnboardingHost` also suppresses the modal, without writing that flag, when any active Cloud backend's settings report a usable LLM (non-empty model plus API-key or subscription auth); this readiness exception must not apply to Local backends. It tracks logical phases (`backend`, `agent`, `setup`, `hello`) rather than fixed numeric steps; the backend phase may be omitted for an already configured backend. Agent choices are derived from `ACP_PROVIDERS` plus OpenHands. The setup phase renders `SetupLlmStep` for OpenHands or `SetupAcpSecretsStep` for ACP providers. Keep the phase-based navigation so adding or removing the backend slide cannot move users to the wrong step.

- Files tab diff-view default logic: keyed off `useHasAttachedSource()` (`src/hooks/use-has-attached-source.ts`), which is true when the user explicitly attached _either_ a repo (`conversation.selected_repository`) _or_ a local workspace (`getStoredConversationMetadata(id).selected_workspace`, persisted by `createConversation` when `workingDirOverride` is supplied). The agent-server pre-initialises every conversation workspace as a git worktree for its own change tracking, so do NOT use a filesystem probe (`git status` / `useUnifiedGetGitChanges`) as the attachment signal — that was tried in earlier iterations and made every fresh no-attachment conversation incorrectly default to diff view. The companion `useHasGitCommits` probe (`src/hooks/query/use-has-git-commits.ts`) then suppresses diff view for attached-but-empty cases (unborn HEAD, non-git workspace).

- Collapsible thinking: `ThinkAction` events and LLM extended reasoning (`reasoning_content` / `thinking_blocks` on `ActionEvent`) are rendered as collapsible sections via `CollapsibleThinking` (`src/components/conversation-events/chat/event-message-components/collapsible-thinking.tsx`). Collapsed by default to keep the chat compact — the thinking is often in English regardless of the user's conversation language. The `getReasoningContent()` helper in `event-thought-helpers.ts` extracts the content, preferring `reasoning_content` (plain string) and falling back to Anthropic `thinking_blocks`. i18n keys: `THINKING$TITLE`, `THINKING$EXPAND`, `THINKING$COLLAPSE`. Tests: `__tests__/components/conversation-events/chat/event-message-think-action.test.tsx`.

- Agent delegation settings: the `Settings > Agent` page (`src/routes/agent-settings.tsx`) is intentionally NOT a `SdkSectionPage` wrapper. It mirrors upstream OpenHands#14418 — it flatMaps every section of `agent_settings_schema` and finds the `enable_sub_agents` field by key, so it works regardless of which section the real backend exposes the field in. Don't refactor it back to `SdkSectionPage` unless you also know the real backend's section name and add a fallback for the live "SDK schema unavailable" path. The toggle persists via `agent_settings_diff`. Nav item lives in `OSS_NAV_ITEMS` (settings-nav.tsx) with the robot icon (`SETTINGS$NAV_AGENT`). The mock schema in `settings-handlers.ts` puts the field in a `general` section. **Client-side gate**: `getAgentTools()` in `agent-server-adapter.ts` only attaches `task_tool_set` to new conversations when `agent_settings.enable_sub_agents === true`. Without that gate the agent server would still receive the tool whenever it advertised it in `/api/server_info`, so the toggle had no effect on running conversations.

- Settings naming is backend-aware today: local `/settings` is profile-oriented (`use-settings-nav-items.ts` renames the first settings item/title/subtitle to `LLM Profiles` and `chat-input-model.tsx` / `chat-input-actions.tsx` link there as `LLM Profiles`), while cloud keeps the generic `LLM Settings` copy because cloud still edits raw settings rather than saved profiles. The local profile editor (`llm-settings-local-view.tsx`) should keep explicit create/edit profile headings plus helper text so users know they are saving a profile, not mutating the current conversation directly.

- Cloud conversation resume gating: when a cloud conversation is closed from the UI (`pauseCloudSandbox` is called), the conversation's `conversation_url` is NOT cleared -- it still points to the old sandbox host. `WebSocketProviderWrapper` must suppress the URL (pass `null` to `ConversationWebSocketProvider`) while `sandbox_status === "PAUSED"`, otherwise the WebSocket immediately tries the stale URL before the sandbox wakes. Symmetrically, `useActiveConversation`'s refetch interval must fast-poll (3 s) on both `!conversation_url` AND `sandbox_status === "PAUSED"` -- checking only the missing URL would leave the hook on the 30 s interval while the sandbox is resuming. The resume sequence: navigate -> sandbox PAUSED detected -> `resumeCloudSandbox` called (in `conversation.tsx`) -> fast-poll detects RUNNING -> `conversationUrl` unblocked -> WebSocket connects.
