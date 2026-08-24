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
- [`OpenHands/extensions`](https://github.com/OpenHands/extensions) owns reusable skills, plugins, automations, and integrations.
- [`OpenHands/extensions`](https://github.com/OpenHands/extensions) owns reusable skills, plugins, automations, and integrations; [`OpenHands/automation`](https://github.com/OpenHands/automation) owns automation definitions, scheduling, webhooks, run history, and dispatching; Agent Server/SDK code executes the dispatched conversations.

When a feature crosses repositories, implement the backend contract in the SDK first, expose it through `typescript-client`, and consume it in Canvas. Coordinate automation lifecycle changes in `automation`. See the repository [contributor notes](../AGENTS.md) and follow the [custom code-review guide](../.agents/skills/custom-codereview-guide.md) for every pull request.

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

The dev stack uses `uvx` to run a temporary `agent-server` installation on
`127.0.0.1:18000` and points the frontend at it. By default, development state
is persisted under `~/.openhands/agent-canvas`; set `OH_CANVAS_SAFE_STATE_DIR`
to choose another directory. The launcher keeps conversation persistence,
workspaces, bash events, secrets, and tmux sockets together there so an
interrupted launcher can be restarted without deleting conversation state. If
`$HOME` is on a filesystem that does not support Unix domain sockets (some
devcontainers, NFS/CIFS homes), set the standard `TMUX_TMPDIR` env var to a
local path such as `/tmp` and the dev stack will use it instead.

### Environment Variables

| Variable                         | Description                                    | Default                     |
| -------------------------------- | ---------------------------------------------- | --------------------------- |
| `PORT`                           | Ingress port for the full/static launcher      | `8000`                      |
| `OH_CANVAS_SAFE_BACKEND_PORT`    | Agent-server port for full/minimal development | `18000`                     |
| `OH_CANVAS_SAFE_AUTOMATION_PORT` | Automation backend port for full/static mode   | `18001`                     |
| `OH_CANVAS_SAFE_VITE_PORT`       | Vite/static frontend port for full/static mode | `3001`                      |
| `OH_CANVAS_SAFE_VSCODE_PORT`     | Editor sidecar port in minimal mode            | `backend + 1`               |
| `OH_CANVAS_SAFE_STATE_DIR`       | Base directory for isolated development state  | `~/.openhands/agent-canvas` |
| `OH_CANVAS_EXTRA_BACKEND_PORT`   | Extra backend port for shared-state mode       | `18002`                     |
| `OH_CANVAS_EXTRA_VSCODE_PORT`    | Extra editor port for shared-state mode        | `18003`                     |
| `OH_AUTOMATION_GIT_REF`          | Git ref for automation backend                 | `main`                      |
| `OH_AGENT_SERVER_GIT_REF`        | Git ref for agent-server                       | `main`                      |
| `VITE_FRONTEND_PORT`             | Frontend port for direct Vite development      | `3001`                      |

### Alternative: Minimal Mode (without Automation)

To run without the automation service:

```sh
npm run dev:minimal
```

This runs only agent-server + Vite (no automation backend or ingress).
Access at `http://localhost:3001/`

### Agent server version selection

By default, the latest released version from PyPI is used. You can override this (highest precedence first):

```sh
# Run against a local software-agent-sdk checkout.
OH_AGENT_SERVER_LOCAL_PATH=/abs/path/to/software-agent-sdk npm run dev

# Use a git branch or commit (takes precedence over version)
OH_AGENT_SERVER_GIT_REF=main npm run dev
OH_AGENT_SERVER_GIT_REF=abc1234 npm run dev

# Use a specific PyPI version
OH_AGENT_SERVER_VERSION=1.18.0 npm run dev
```

`OH_AGENT_SERVER_LOCAL_PATH` must be an absolute path to a `software-agent-sdk` checkout containing the `openhands-agent-server`, `openhands-sdk`, `openhands-tools`, and `openhands-workspace` workspace packages. The agent-server itself is rebuilt from local source on each start (`uvx --reinstall`); the other workspace packages are installed editable, so their source changes take effect without a rebuild.

### Other useful overrides

- `OH_CANVAS_SAFE_BACKEND_PORT` — backend port for the isolated server (default `18000`)
- `OH_CANVAS_SAFE_VSCODE_PORT` — VS Code sidecar port (default `backend port + 1`)
- `OH_CANVAS_SAFE_STATE_DIR` — base directory for isolated server state
- `VITE_WORKING_DIR` — repo root used for new conversations (defaults to the current checkout)

## Port allocation and collision recovery

The launcher checks its preferred ports before starting services and fails fast
when a required port is already in use. The values below are the defaults; the
full and static launchers accept `PORT`,
`OH_CANVAS_SAFE_BACKEND_PORT`, `OH_CANVAS_SAFE_AUTOMATION_PORT`, and
`OH_CANVAS_SAFE_VITE_PORT` overrides. In full mode the editor port is derived
from the agent-server port (`backend + 1000`); in minimal mode it is
`backend + 1` and can be overridden with `OH_CANVAS_SAFE_VSCODE_PORT`.

| Development mode | Entry point                                                        | Ports owned by the launcher                                                                                                                             |
| ---------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full             | `npm run dev`                                                      | Ingress `8000`; agent-server `18000`; automation `18001`; Vite `3001`; editor `19000` (`18000 + 1000`, advertised through the ingress `/vscode/` route) |
| Static frontend  | `npm run dev:static`                                               | Same ingress, agent-server, automation, and editor ports as full mode; the built frontend is served on the frontend port `3001` instead of by Vite      |
| Minimal          | `npm run dev:minimal`                                              | Frontend `3001`; agent-server `18000`; editor `18001`; no automation backend or ingress                                                                 |
| Frontend-only    | `agent-canvas --frontend-only` or `npm run dev -- --frontend-only` | Ingress `8000` and frontend `3001`; no agent-server, automation backend, or editor; point the frontend at an existing backend when needed               |
| Backend-only     | `agent-canvas --backend-only` or `npm run dev -- --backend-only`   | Ingress `8000`; agent-server `18000`; automation `18001`; editor `19000`; no frontend server                                                            |
| Mock frontend    | `npm run dev:mock`                                                 | Frontend `3001` only; no live backend is managed                                                                                                        |
| Extra backend    | `npm run dev:extra-backend`                                        | Agent-server `18002` and editor `18003`; shares the default development state and is not an isolated second stack                                       |

The frontend-only, backend-only, and static modes use the full launcher's
ingress when one is started. The ingress only routes to services enabled by
that mode. If a mode is changed to use a different backend port, update the
corresponding frontend/backend URL as well instead of assuming that `8000` is
still the target.

### Diagnose a port collision without deleting state

First stop the launcher from its owning terminal with `Ctrl-C`. If a child
process remains, identify the listener before stopping anything. Do not kill
all `node`, `python`, or `uvx` processes: another OpenHands stack may own them.

On macOS, use `lsof`; on Linux, use `ss` (or `lsof` if it is installed):

```sh
# macOS
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:18000 -sTCP:LISTEN

# Linux; replace the port list with the mode you are running
ss -ltnp | grep -E ':(8000|18000|18001|19000|3001)\b'
```

On Windows PowerShell, query the listeners and then inspect the owning PID:

```powershell
$ports = 8000, 18000, 18001, 19000, 3001
Get-NetTCPConnection -State Listen |
  Where-Object { $_.LocalPort -in $ports } |
  Select-Object LocalAddress, LocalPort, OwningProcess
Get-Process -Id <PID>
```

After identifying the process, stop only that process gracefully and rerun the
listener query. Use `kill -TERM <PID>` on macOS/Linux or
`Stop-Process -Id <PID>` in PowerShell. Prefer `Ctrl-C` or a normal process
stop; do not use `kill -9`, `Stop-Process -Force`, or destructive state cleanup
unless you have separately established that the process is stuck and the
state is backed up.

### Start two isolated full stacks

Give the second stack its own ingress, service ports, and state directory. The
state directory is the isolation boundary; changing ports alone does not
prevent two stacks from sharing conversations and secrets.

```sh
# Terminal 1: defaults
npm run dev

# Terminal 2: a separate stack; the editor port is derived as 19100
PORT=8100 \
OH_CANVAS_SAFE_BACKEND_PORT=18100 \
OH_CANVAS_SAFE_AUTOMATION_PORT=18101 \
OH_CANVAS_SAFE_VITE_PORT=3101 \
OH_CANVAS_SAFE_STATE_DIR="$HOME/.openhands/agent-canvas-2" \
npm run dev
```

The second stack is available at `http://localhost:8100/`. In Windows
PowerShell, set the same variables before launching:

```powershell
$env:PORT = '8100'
$env:OH_CANVAS_SAFE_BACKEND_PORT = '18100'
$env:OH_CANVAS_SAFE_AUTOMATION_PORT = '18101'
$env:OH_CANVAS_SAFE_VITE_PORT = '3101'
$env:OH_CANVAS_SAFE_STATE_DIR = "$HOME\.openhands\agent-canvas-2"
npm run dev
```

Keep the default state directory intact when recovering a stack. To reset or
discard development state, make that a deliberate, separate operation after
backing up any conversations you need; a port collision by itself is not a
reason to remove it.

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
