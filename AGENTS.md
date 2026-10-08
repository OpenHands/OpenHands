# Repository Notes

## General

- This repository is the OpenHands Agent Canvas React/TypeScript frontend.

## i18n workflow
- After editing `src/i18n/translation.json`, run `npm run make-i18n` to
  regenerate `src/i18n/declaration.ts` (gitignored, auto-generated) and
  `scripts/check-translation-completeness.cjs` to confirm full locale coverage.
- Removing an i18n key: delete the whole locale block from translation.json,
  then `make-i18n`, then remove any test-mock entries (tests inline a
  `translations` map in the `useTranslation` vi.mock).

## Commit messages
- The persistent shell can garble long multi-line `commit -m` bodies (echo
  loop corrupts the stored message). Use `git commit -F <file>` with a
  written temp file instead of `-m` for any non-trivial message.

- Primary verification commands are `npm run lint`, `npm test`, `npm run build`, and `npm run build:lib`.
- Direct dependencies and dev dependencies are exact-pinned. Use the committed `package-lock.json` with `npm ci`, and update `package.json` and `package-lock.json` together through npm.
- Public skills come from `@openhands/extensions`; project-specific contributor guidance lives under `.agents/skills/`.
- Default working-directory behavior must reuse `DEFAULT_WORKING_DIR` from `src/api/agent-server-config.ts` rather than hardcoding `/workspace/project`.
- All pull requests must comply with [`.agents/skills/custom-codereview-guide.md`](.agents/skills/custom-codereview-guide.md), in addition to the general contribution requirements and CI checks.

## Repository Ownership

Before adding code, confirm that the change belongs in this repository.

| Repository | Owns | Add code there when… |
|---|---|---|
| [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands) | Agent Canvas UI, frontend state, backend selection, frontend service integration, and local-stack orchestration. | Changing UI, frontend state, or how Canvas consumes an existing API. |
| [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) | Python SDK, Agent Server, agents/tools, conversations, events, canonical REST/WebSocket API, and `clients/typescript/`. | Adding or changing backend behavior, endpoints, wire contracts, or typed client access. |
| [`OpenHands/extensions`](https://github.com/OpenHands/extensions) | Reusable public skills, automations, plugins, and integrations. | Adding or editing reusable extension content rather than Canvas behavior. |
| [`OpenHands/automation`](https://github.com/OpenHands/automation) | Automation definitions, scheduling, webhooks, run history, and dispatch. | Changing when automations run or their scheduling/webhook lifecycle. |

The normal dependency direction is Agent Server contract → TypeScript client → Agent Canvas. Do not reimplement Agent Server endpoints or contracts in Canvas.

## Shared Frontend Change Checklist

Before changing a shared adapter, conversation builder, setting, state selector, or presentation helper:

- Raise `compatibility.minimumAgentServer` in `config/defaults.json` when Canvas begins requiring a new Agent Server endpoint, field, schema, or behavior.
- Enumerate affected consumers across Local and Cloud backends, OpenHands and ACP agents, standard/planning/delegated conversations, and standalone and embedded Canvas. Test every path with distinct control or data flow without testing an irrelevant Cartesian product.
- Give durable frontend state one named owner and one obvious writer. Handle existing-state hydration or migration and all supported create, update, delete, default, and active-item transitions. Leave destructive transitions in a deterministic valid state or an explicit empty state handled by the UI.

## PR Description Human Check

The `HUMAN:` section in PR descriptions is reserved for human contributors only. AI agents must not add to, edit, move, or remove it. If validation fails because it is missing or empty, stop and ask the human user to update it in their own words. If a human already updated it, report the exact validator error rather than editing it.

## Functionality-Specific Skills

Detailed contributor knowledge is split into skills so it loads only for relevant work. Invoke every applicable skill before changing that area; cross-cutting changes may require several skills.

| Skill | Use for |
|---|---|
| [`telemetry-analytics`](.agents/skills/telemetry-analytics/SKILL.md) | PostHog, telemetry consent/identity, typed tracking events, Cloud funnel observability, and onboarding instrumentation. |
| [`e2e-testing`](.agents/skills/e2e-testing/SKILL.md) | Live or mock-LLM Playwright suites, Docker E2E, E2E CI/reporting, artifacts, and failure diagnosis. |
| [`frontend-api-contracts`](.agents/skills/frontend-api-contracts/SKILL.md) | `src/api`, typed Agent Server calls, Cloud/runtime transport, compatibility, backends, settings, secrets, auth, and conversation contracts. |
| [`local-stack-runtime`](.agents/skills/local-stack-runtime/SKILL.md) | Dev launchers, runtime-services metadata, ingress, automation startup, process shutdown, centralized versions, and Docker. |
| [`desktop-electron`](.agents/skills/desktop-electron/SKILL.md) | Electron startup, branding, packaging, bundled Node/uv, universal macOS builds, and desktop CI. |
| [`frontend-development`](.agents/skills/frontend-development/SKILL.md) | React/UI work, i18n, named identifiers, MSW mock mode, lazy loading, bundle performance, and feature-specific UI invariants. |
| [`pr-design-doc`](.agents/skills/pr-design-doc/SKILL.md) | Writing the PR design document stored in the PR body. |
| [`release`](.agents/skills/release.md) | Cutting and verifying an `@openhands/agent-canvas` release. |
| [`verify-openhands`](.agents/skills/verify-openhands/SKILL.md) | Driving the real app like a user with `control-openhands`, the feature map of every user-facing behavior, and creating or maintaining that map. |

The detailed rules live in each skill's `references/guide.md`; do not copy them back into this file. Update the owning skill whenever an invariant changes.

## Testing Baseline

Create TDD tests for behavioral changes. Keep tests focused on user behavior and real code paths:

- Use Arrange, Act, Assert structure and clear test data.
- Avoid duplicate cases and duplicate assertions.
- Mock an underlying service rather than the hook that consumes it.
- Extend an existing test file when it is a natural home; create a new file only when necessary.
- Avoid brittle presentation-only assertions. Test functional CSS contracts directly when they are behavior.
- Do not mirror literal source, fixture, translation, or class-string definitions in tests. Assert consumer behavior or an actual build/runtime contract instead, and do not export internals solely to make them testable. Preserve coverage for routing, accessibility, async behavior, behavioral contracts, and functional CSS/build contracts.
- Use the minimum number of cases that fully cover the intended behavior and edge cases.

Use the `e2e-testing` skill for suite selection and E2E-specific requirements.

## Specifications and Releases

- Spec files live under `specs/`. Keep spec IDs stable and mark deprecated specs with strikethrough rather than renumbering them.
- Tag implementation code and tests with `// @spec BM-002 — Short title` comments immediately above the relevant block so coverage remains grep-able.
- Follow `.agents/skills/release.md` for release automation. Never hand-edit a release-please branch.

## DigitalOcean Managed Agents (Electron only)

- **Surfaces**: the "Choose how you want to connect" chooser (`AddBackendChooser` in `backend-form-modal.tsx`) gains a third **DigitalOcean** tab, `DigitalOceanConnectPanel`: sign in (OAuth or token), then pick an OpenHands agent — opening it resumes the agent's latest live session or launches its first, and closes the modal. Each agent row expands to list its sessions as children (any connectable one opens directly), and **New Agent** (`DigitalOceanNewAgentForm`) creates an OpenHands Agent Config and launches its first session. While a MARS backend is active, `MarsSessionPanel` (`src/components/features/sidebar/mars-session-panel.tsx`) replaces the sidebar conversation list: the agent's sessions read like thread folders (the current one expands to its conversations, others open with a click), the header "+" launches a new session, and the agent menu switches agent or opens `ManageBackendsModal initialView="managed-agents"` — the full management view in `src/components/features/backends/managed-agents/`. Everything is gated on `getMarsBridge()`: Electron's preload provides it, and the web build gets a fetch-backed one from the Agent Canvas server when that server hosts the MARS web bridge (below); a page with neither never shows it.
- **OpenHands only**: agent configs are shown only when their manifest's `agent` is `openhands` (flat `agent:` or `spec.agent`). The list endpoint omits manifests, so `listAgentConfigs` in `scripts/mars-tunnel-bridge.mjs` fetches each config once (configs are immutable) and adds `agent`. The match is strict: a manifest without `agent: openhands` (older specs that no longer launch) or one that could not be read (`agent: null`, retried on the next list) is hidden. Sessions are shown only when their `config_id` names a matching config; `agent_kind` is not used, since OpenHands sessions report `AGENT_KIND_UNSPECIFIED`.
- **New Agent**: `createOpenHandsAgent({name, llmApiKey?})` in `scripts/mars-tunnel-bridge.mjs` posts `POST /v2/agents/configs` `{name, manifest_yaml}` with the manifest from `buildOpenHandsManifest()` (`scripts/mars-api.mjs`): flat `agent: openhands` + `template: openhands`, plus `secrets.OPENHANDS_LLM_API_KEY.value` when a key is given. The manifest is built in the main process, not accepted over IPC, so the renderer can only create OpenHands agents (same reasoning as pinning the guest port). Keep `template` explicit — harness-api has no agent kind that derives an OpenHands template. Names follow harness-api's rule (`isValidAgentName`: 1–64 of `[A-Za-z0-9._-]`, alphanumeric at both ends).
- **Shared hooks**: `useLaunchMarsSession()` (open / launch / create agent / "open agent", with create-vs-connect error copy) and `useMarsSignIn()` back the chooser tab and the sidebar; reuse them instead of calling `createSession` + connect directly.
- **Tunnel first, public URL as fallback**: `openTunnel` opens the port-forward tunnel (`{transport: "tunnel", host: http://127.0.0.1:<port>}`) and, only when that cannot be opened, resolves the session's public URL from harness-api (`GET /v2/agents/sessions/{id}/ingress`, `scripts/mars-ingress.mjs` — waking a paused session first, polling PENDING → READY) and returns `{transport: "ingress", host}`. The tunnel is the default because it is the path proven end to end today, WebSocket included, on desktop and on the web build; the public URL is the intended path once its WebSocket hop (microVM LB → activator) is fixed, and promoting it is just flipping this order. The hostname is revoked on pause and lock and changes after rollback, so it is re-resolved on every connect and restore, never reused from the persisted backend.
- **Bearer injection, not a proxy**: the ingress URL is PAT-authenticated on every request, including the WebSocket handshake, and the renderer cannot set WebSocket headers nor hold the token. `registerRequestAuth(session.defaultSession)` installs a `webRequest.onBeforeSendHeaders` hook (filter `https://*/*`, `wss://*/*`) that stamps `Authorization: Bearer <token>` onto the renderer's requests whose host is a connected session's ingress host, reading the owning connection's token live so sign-out cuts it off. Nothing else is touched. The event WebSocket's host comes from the agent-server's `conversation_url`, so the gateway must forward `Host` for that URL to carry the public hostname.
- **Web build (browser)**: the server that fronts the web build (`scripts/ingress.mjs` for `npm run dev`, `scripts/static-server.mjs` for `dev:static` and Docker) hosts the same bridge via `scripts/mars-web-bridge.mjs` and exposes it same-origin: `POST /mars/rpc/<method>` (the `MarsBridge` surface; `src/api/mars/mars-web-bridge.ts` installs a fetch-backed `window.marsBridge` from `entry.client.tsx` after probing `GET /mars/health`), and `/mars/sessions/<id>/…` as an HTTP + WebSocket reverse proxy to that session's connected host with `Authorization: Bearer` added server-side and the upstream socket pinged every 30 s (the ingress route idles out at 20 min). `openTunnel` therefore answers `host: <origin>/mars/sessions/<id>`, a path-prefixed backend; proxied JSON has agent-server URLs (`conversation_url` and friends) rewritten to that prefix so the event WebSocket lands on the proxy. This is what makes the browser work at all: harness-api has no CORS, a browser WebSocket cannot carry the bearer, and the token must stay off the page. The routes are **off unless `MARS_WEB=1`** and, when on, want the server's session key on every request: the renderer sends it as `X-Session-API-Key` on `/mars/rpc/*` and gets an `HttpOnly; SameSite=Strict; Path=/mars` cookie back, which the browser attaches on its own to the proxied REST calls and the WebSocket upgrade (the one request a browser cannot put a header on). The key is `MARS_WEB_KEY`, else the server's `--session-api-key` (`dev-with-automation` hands the stack's key to `ingress.mjs`); with neither, `mountMarsWebBridge` mounts only on a loopback bind, where a `Host` check stops DNS rebinding. Only `conversation_url` fields are rewritten (a URL inside a message is content), JSON past 8 MB streams through untouched, `Cookie` and the key header never reach the session host, a malformed session path is a 400, and an upstream WebSocket the browser abandoned mid-handshake is terminated. A 404 from `/ingress` (harness-api without it yet) and a URL that never becomes READY both fall back to the tunnel like a 501. Credentials are in-memory (no keychain; `MARS_TOKEN` seeds one at startup), OAuth is not offered (`canUseOAuth: false`), and cross-origin requests are refused.
- **Stop and delete are confirmed, destructive, and gated**: Stop (`destroySession`) ends the session and its sandbox through `DELETE /v2/agents/sessions/{id}`, detaching any live connection first; Delete agent (`deleteAgentConfig`) soft-deletes the config through `DELETE /v2/agents/configs/{id}`. harness-api does not refuse a delete while sessions run, so the renderer only enables it once none of the agent's sessions is live. Both sit behind `ConfirmationModal` and ride the same IPC / web-RPC surface as pause.
- **One backend per session**: connecting registers an ordinary `kind: "local"` backend whose `host` is the public ingress URL (or the loopback tunnel; in the web build, the server's `/mars/sessions/<id>` proxy), carrying `marsSessionId` / `marsConfigId`. The far end is a real agent-server, so conversations, files, and the terminal work unmodified. Connect logic is shared through `useConnectMarsSession()` (`src/hooks/use-connect-mars-session.ts`), which lands the user in the session's most recent conversation (or a new chat) with the URL pinned to the new backend.
- **Main-process ownership**: ingress resolution, tunnels, the harness-api REST client, OAuth, and credentials live in the Electron main process (`scripts/mars-tunnel-bridge.mjs`, `mars-ingress.mjs`, `tunnel-client.mjs`, `tunnel-registry.mjs`, `mars-api.mjs`, `mars-oauth.mjs`, `mars-credentials.mjs`) and reach the renderer only via `window.marsBridge` (`electron/preload-main.cjs`). harness-api sends no CORS headers and tokens must never cross the context bridge, so do not move these calls into the renderer.
- **Fail fast, explain why**: the tunnel records the last upstream failure (`getTunnel().upstreamFailure`). `waitForMarsAgentServer` throws `MarsAttachError("refused")` immediately on close code 4001 or HTTP 401/403, and `MarsAttachError("not-openhands")` when close code 4002 (guest dial failed — nothing on :8000) outlasts a 20 s grace window, since a just-woken sandbox also closes 4002 while booting. `getMarsConnectErrorMessage` maps both to `DO_AGENTS$ERROR_*` copy.
- **Cost**: every probe and tunnel dial counts as sandbox activity and keeps it from auto-pausing. `useRestoreMarsTunnels` reopens only the *active* backend's tunnel, and `useBackendsHealth` probes non-active MARS backends once without interval/focus/reconnect refetches. Do not add background polling of sessions the user is not on.
- **Auth**: OAuth uses DigitalOcean's implicit grant (no client secret can ship in a desktop binary) with a fixed loopback redirect `http://127.0.0.1:53682/callback`; personal access tokens always work as a fallback. The public client ID comes from `DO_OAUTH_CLIENT_ID`, else `mars.oauthClientId` in `config/defaults.json` (empty = PAT only); `MARS_API_BASE_URL` / `mars.apiBaseUrl` select the harness API. Credentials are `safeStorage`-encrypted and session-only when the OS keychain is unavailable.
- **Tests**: `__tests__/components/backends/managed-agents-view.test.tsx`, the "DigitalOcean" block of `__tests__/components/backends/add-backend-modal.test.tsx`, `__tests__/components/features/sidebar/mars-session-panel.test.tsx`, `__tests__/hooks/use-mars-tunnel-backend.test.tsx`, and `__tests__/scripts/{mars-*,tunnel-*}.test.ts`. Component tests fake `window.marsBridge`, not the hooks.
