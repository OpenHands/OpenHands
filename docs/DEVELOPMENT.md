# Development

This document is for contributors working on `agent-canvas` itself.

## Recommended local workflow

`npm run dev` runs the full local stack (agent-server + automation backend via
`uvx`, Vite dev server with live reload, and an ingress proxy) — all without
Docker.

## Repository boundaries

This repository contains the Agent Canvas frontend and local-stack orchestration. Use the sibling repositories for their owned layers:

- [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) owns the Python SDK, Agent Server, agent/tool behavior, conversations, workspaces, events, and server API.
- [`OpenHands/typescript-client`](https://github.com/OpenHands/typescript-client) owns browser-compatible typed access to that Agent Server API. Add client methods there rather than reimplementing API calls in Canvas.
- [`OpenHands/extensions`](https://github.com/OpenHands/extensions) owns reusable skills, plugins, automations, and integrations; [`OpenHands/automation`](https://github.com/OpenHands/automation) owns automation definitions, scheduling, webhooks, run history, and dispatching; Agent Server/SDK code executes the dispatched conversations.

When a feature crosses repositories, implement the backend contract in the SDK first, expose it through `typescript-client`, and consume it in Canvas. Coordinate automation lifecycle changes in `automation`. Version pins, local-stack overrides, and PR ordering are in [Cross-repository version compatibility](#cross-repository-version-compatibility). See the repository [contributor notes](../AGENTS.md) and follow the [custom code-review guide](../.agents/skills/custom-codereview-guide.md) for every pull request.

For a static frontend build (better for slow networks, remote access, tunnels):

```sh
npm run dev:static
```

The published `agent-canvas` binary also supports partial-stack modes when you want to run the frontend and backend processes separately:

```sh
agent-canvas --frontend-only
agent-canvas --backend-only
```

Both modes still start the ingress proxy; the proxy only routes to the services started by that mode.

The dev stack uses `uvx` to run a temporary `agent-server`
installation on `127.0.0.1:18000` and points the frontend at it. It isolates
conversation persistence by setting separate `OH_CONVERSATIONS_PATH`,
`OH_BASH_EVENTS_DIR`, and `OH_VSCODE_PORT` values under `.openhands-dev/`, and
keeps its tmux sockets under `~/.openhands/agent-canvas/tmux` (via
`TMUX_TMPDIR`), so it does not collide with other local or cloud-backed
OpenHands sessions. If `$HOME` is on a filesystem that does not support Unix
domain sockets (some devcontainers, NFS/CIFS homes), set the standard
`TMUX_TMPDIR` env var to a local path such as `/tmp` and the dev stack will use
it instead.

### Environment Variables

| Variable                     | Description                            | Default                        |
| ---------------------------- | -------------------------------------- | ------------------------------ |
| `PORT`                       | Ingress port                           | `8000`                         |
| `OH_AGENT_SERVER_LOCAL_PATH` | Absolute `software-agent-sdk` checkout | unset                          |
| `OH_AGENT_SERVER_GIT_REF`    | Git ref for agent-server               | unset (`config/defaults.json`) |
| `OH_AGENT_SERVER_VERSION`    | PyPI version for agent-server          | unset (`config/defaults.json`) |
| `OH_AUTOMATION_LOCAL_PATH`   | Absolute `automation` checkout         | unset                          |
| `OH_AUTOMATION_GIT_REF`      | Git ref for automation backend         | unset (`config/defaults.json`) |
| `OH_AUTOMATION_VERSION`      | PyPI version for automation            | unset (`config/defaults.json`) |

### Alternative: Minimal Mode (without Automation)

To run without the automation service:

```sh
npm run dev:minimal
```

This runs only agent-server + Vite (no automation backend or ingress).
Access at `http://localhost:3001/`

### Cross-repository version compatibility

The default local stack for this checkout is the **pins recorded below**, not
an open-ended matrix of sibling `main` branches. Connecting to another Agent
Server is allowed down to `compatibility.minimumAgentServer`. Mixing an
unpinned Git backend with the committed TypeScript client (or the reverse) is
unsupported and often looks like a product bug.

Owning repositories:

- [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands) (this repo) — Agent Canvas
- [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) — Python SDK and Agent Server
- [`OpenHands/typescript-client`](https://github.com/OpenHands/typescript-client) — `@openhands/typescript-client`
- [`OpenHands/automation`](https://github.com/OpenHands/automation) — scheduling, webhooks, run history
- [`OpenHands/extensions`](https://github.com/OpenHands/extensions) — `@openhands/extensions` skills and integrations

#### Source of truth

| Surface                                    | Supported version lives in                                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| Bundled Agent Server / SDK PyPI pin        | [`config/defaults.json`](../config/defaults.json) `versions.agentServer`      |
| Bundled automation PyPI pin                | `config/defaults.json` `versions.automation`                                  |
| Oldest Agent Server this frontend accepts  | `config/defaults.json` `compatibility.minimumAgentServer`                     |
| `@openhands/typescript-client`             | [`package.json`](../package.json) (exact npm pin)                             |
| `@openhands/extensions`                    | `package.json` (exact npm pin)                                                |
| Released automation ↔ SDK dependency match | [`scripts/check-sdk-version-sync.mjs`](../scripts/check-sdk-version-sync.mjs) |

`config/defaults.json` is the source of truth for the Python backend pins used
by the npm and Docker install paths. `agent-canvas --info` prints the same
Agent Server and automation pins plus the minimum compatible Agent Server.

Those files describe **this revision**. They are not a historical compatibility
matrix. They do not promise that Git `main` of every sibling repository works
together, or that independently chosen PyPI/npm numbers with the same major
version are interchangeable.

Runtime enforcement for a connected Agent Server is
`assertAgentServerVersionIsSupported()` in
[`src/api/agent-server-compatibility.ts`](../src/api/agent-server-compatibility.ts).
Some UI features have additional floors inside `@openhands/typescript-client`.
Meeting `compatibility.minimumAgentServer` does not mean every Canvas feature
is available on that backend.

The TypeScript client records the Agent Server contract it was generated from
in that repository (see its README, "Agent Server API contract"). That client
contract pin is independent of Canvas `versions.agentServer`.

#### Released package, Git ref, and local path

`npm run dev`, `npm run dev:static`, and the `agent-canvas` binary select
**Python** backends with the following precedence (highest first).
`npm run dev:minimal` uses the same Agent Server selection and does not start
automation. Leaving an override unset means "use the pin above", not Git
`main`.

**Agent Server** ([`software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk)):

1. `OH_AGENT_SERVER_LOCAL_PATH` — absolute path to a checkout that contains
   `openhands-agent-server`, `openhands-sdk`, `openhands-tools`, and
   `openhands-workspace`. The agent-server package is rebuilt from local
   source on each start (`uvx --reinstall`); the other workspace packages are
   installed editable.
2. `OH_AGENT_SERVER_GIT_REF` — branch, tag, or commit. All four workspace
   packages are installed from that same ref so inter-package APIs stay in
   sync. The launcher passes `uvx --reinstall` so a cached PyPI wheel with
   the same version string is not reused.
3. `OH_AGENT_SERVER_VERSION` — a specific PyPI version of those four packages.
4. Default: `versions.agentServer` from `config/defaults.json`.

```sh
OH_AGENT_SERVER_LOCAL_PATH=/abs/path/to/software-agent-sdk npm run dev
OH_AGENT_SERVER_GIT_REF=<branch-or-sha> npm run dev
OH_AGENT_SERVER_VERSION=<versions.agentServer> npm run dev
```

**Automation** ([`automation`](https://github.com/OpenHands/automation)):

1. `OH_AUTOMATION_LOCAL_PATH` — absolute path to a checkout with
   `pyproject.toml`. `--automation-ref` on the launcher outranks this local
   path.
2. `OH_AUTOMATION_GIT_REF` (or `--automation-ref`) — branch, tag, or commit.
   `OH_AUTOMATION_REPO` only applies when a git ref is selected.
3. `OH_AUTOMATION_VERSION` — a specific PyPI version of `openhands-automation`.
4. Default: `versions.automation` from `config/defaults.json`.

Released `openhands-automation` is checked against `versions.agentServer`.
CI runs `scripts/check-sdk-version-sync.mjs` on the **published** automation
package, not on a local checkout or Git `main`.

**TypeScript client and extensions:** Canvas does not provide Git-ref or
local-path launcher variables for `@openhands/typescript-client` or
`@openhands/extensions`. Both are exact npm pins in `package.json`. Public
skills are loaded from `@openhands/extensions` at **build time**; the Agent
Server no longer clones the extensions repo or honors `EXTENSIONS_REF`.

Contributor notes require a **published** TypeScript client before Canvas
bumps that pin. Do not point this repository at an unpublished commit SHA.

Iterate on unreleased client or extensions work in those repositories, then
bump the Canvas pin after the package exists on the registry.

#### Cross-repository change checklist

1. Confirm ownership using [Repository boundaries](#repository-boundaries).
2. Read `config/defaults.json` and `package.json` for the pins this Canvas
   revision expects.
3. If the Agent Server API changes, land it in `software-agent-sdk` first.
   For local Canvas testing, use `OH_AGENT_SERVER_LOCAL_PATH` or
   `OH_AGENT_SERVER_GIT_REF`; keep the TypeScript client pin until the client
   is published.
4. Mirror the contract in `typescript-client` (OpenAPI and handwritten
   clients). Publish that package, then bump `@openhands/typescript-client`
   here.
5. If automation must run against the new SDK, release `openhands-automation`
   with matching SDK dependencies, then update `versions.automation` so
   `scripts/check-sdk-version-sync.mjs` still passes.
6. If public skills or integrations change, publish `@openhands/extensions`
   and then bump that dependency.
7. If a Canvas PR needs an unreleased Agent Server, link the
   `OpenHands/software-agent-sdk` **pull request** (not only an issue) in the
   Canvas PR body so
   [mock-LLM Docker e2e](../.github/workflows/mock-llm-docker-e2e.yml) can
   install that git ref.
8. Do not merge Canvas UI that requires a contract the pinned client does not
   yet expose.

#### Breaking contract sequencing

Usual direction: Agent Server / SDK → OpenAPI contract → `typescript-client` →
Agent Canvas. Automation scheduling flows Canvas → `automation` → Agent
Server / SDK.

REST deprecation and the removal runway are owned by
`software-agent-sdk` (`openhands-agent-server/AGENTS.md`). Canvas must not
skip the client release step or consume an unpublished client SHA.

Event wire types follow the same order: SDK Pydantic model, then TypeScript
client, then Canvas consumption of the client type. See the
[custom code-review guide](../.agents/skills/custom-codereview-guide.md).

### Other useful overrides

- `OH_CANVAS_SAFE_BACKEND_PORT` — backend port for the isolated server (default `18000`)
- `OH_CANVAS_SAFE_VSCODE_PORT` — VS Code sidecar port (default `backend port + 1`)
- `OH_CANVAS_SAFE_STATE_DIR` — base directory for isolated server state
- `VITE_WORKING_DIR` — repo root used for new conversations (defaults to the current checkout)

## Alternative development workflows

### Multiple local backends (shared persistence)

To run a second standalone agent-server alongside `npm run dev` while sharing
its conversation history and encrypted secrets, you can use the
`npm run dev:extra-backend` helper. It launches an extra server on `:18002` that
reuses the bundled instance's state dir.

### Frontend against an existing backend

Use this only if you intentionally started `agent-server` yourself or want the frontend to talk to another backend:

```sh
npm run dev:frontend
```

The frontend-only workflow expects the backend at `127.0.0.1:8000` by default.

If you set `LOCAL_BACKEND_API_KEY`, it is used as the API key for the agent-server (mapped internally to `OH_SESSION_API_KEYS_0`). The launcher auto-generates and persists a key when `LOCAL_BACKEND_API_KEY` is not set.

### Mock mode

If you want to run the frontend without a live backend, use:

```sh
npm run dev:mock
```

## Build and test

```sh
npm run test
npm run build
npm run start
```

Useful targeted verification for the isolated dev launcher:

```sh
npm run test -- __tests__/api/agent-server-config.test.ts __tests__/scripts/dev-safe.test.ts
```

### Mutation testing

Stryker checks whether the Vitest suite detects deliberate changes to the
first-party TypeScript source under `src/`. The default configuration excludes
tests, declarations, generated files, fixtures, mocks, and development seeds.

```sh
# Full mutation run (expensive for the whole frontend)
npm run test:mutation

# Reuse results from the previous run
npm run test:mutation:incremental

# Mutate only production files changed from the local main branch
npm run test:mutation:diff

# Compare with another base ref, such as the latest remote main
npm run test:mutation:diff -- origin/main
```

The HTML report is written to `reports/mutation.html`. Mutation scores are
report-only initially; establish a stable baseline before adding a failing
threshold.

Stryker does not cover the small Python surface in this repository; mutating it
would need a Python test harness and Python-specific mutation tool.

## CSS isolation and host-app customization

The standalone app and the exported provider/root wrapper now scope all bundled CSS under a dedicated shell element with the `data-agent-server-ui` attribute. That means Tailwind utilities, HeroUI component styles, xterm styles, and local CSS only apply inside the OpenHands UI subtree instead of leaking into a host app.

### Embedding strategy

- Use `AgentServerUIProviders` in host apps. It renders a scoped style root by default.
- For direct wrapper control, use `AgentServerUIRoot`.
- The standalone app opts out of the provider wrapper because the router layout already renders the scoped root.

### Customization strategy

Theme and surface tokens are exposed as CSS custom properties on the scoped root. You can override them either through the provider/root `styleOverrides` prop or with host CSS targeting `[data-agent-server-ui]`.

```tsx
<AgentServerUIProviders
  styleOverrides={{
    "--oh-color-base": "#101820",
    "--oh-color-content-2": "#f5f7ff",
    "--oh-accent": "#8b5cf6",
  }}
>
  <App />
</AgentServerUIProviders>
```

If you want Tailwind layout utilities on the inner themed container, pass `contentClassName` instead of `className`, because the outer scope element is what all generated selectors key off of.

## Environment variables

You can create a `.env` file in the project directory with these variables based on `.env.sample`.

| Variable                    | Description                                                                               | Default Value          |
| --------------------------- | ----------------------------------------------------------------------------------------- | ---------------------- |
| `VITE_BACKEND_BASE_URL`     | Full base URL for the agent server used by direct browser requests                        | current browser origin |
| `VITE_BACKEND_HOST`         | Backend host used by the Vite dev proxy                                                   | `127.0.0.1:8000`       |
| `VITE_SESSION_API_KEY`      | (Internal) Session API key injected by the launcher — set `LOCAL_BACKEND_API_KEY` instead | -                      |
| `VITE_WORKING_DIR`          | Workspace path sent when starting new conversations                                       | `workspace/project`    |
| `VITE_ENABLE_BROWSER_TOOLS` | Set to `false` to omit `BrowserToolSet` from new conversation payloads                    | `true`                 |
| `VITE_BASE_PATH`            | Build/serve the SPA under a subpath such as `/canvas`                                     | `/`                    |
| `VITE_MOCK_API`             | Enable/disable API mocking with MSW                                                       | `false`                |
| `VITE_USE_TLS`              | Use HTTPS/WSS for the Vite proxy target                                                   | `false`                |
| `VITE_FRONTEND_PORT`        | Port to run the frontend application                                                      | `3001`                 |
| `VITE_INSECURE_SKIP_VERIFY` | Skip TLS certificate verification for proxied backend requests                            | `false`                |
