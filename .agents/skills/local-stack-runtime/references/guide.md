## Deployment Configuration

Supported deployment environment variables:

- `VITE_BACKEND_BASE_URL` — Agent Server base URL.
- `VITE_SESSION_API_KEY` — optional session authentication.
- `VITE_WORKING_DIR` — default workspace path sent when starting conversations.
- `VITE_ENABLE_BROWSER_TOOLS=false` — omit `browser_tool_set` from new conversation payloads.
- `VITE_BASE_PATH` — serve the SPA under a subpath such as `/canvas`; pair it with `scripts/static-server.mjs --base-path` at runtime.

GitHub automation includes `.github/workflows/ci.yml` for `npm ci`, `npm test`, and `npm run build`, plus `.github/dependabot.yml` with weekly npm and GitHub Actions updates gated by a seven-day cooldown.


## Runtime Services in Dev Stacks

- When the agent-canvas dev launchers (`npm run dev` / `dev:static` / the published `agent-canvas` binary) start a stack with ingress/static-server, the backend-facing server appends runtime service metadata to `/server_info` as the optional `runtime_services` field. The frontend reads that backend-provided value when creating conversations and forwards it as `AgentContext.system_message_suffix` on `POST /api/conversations`, so conversations land with a `<RUNTIME_SERVICES>` block appended to the system prompt.
- The block lists URLs **from the agent's point of view**:
  - The Agent Server is always reachable as `http://localhost:<port>` from inside the sandbox — but that is _you_, not the automation backend.
  - Host-side services (ingress, Vite, automation) are reachable as `http://localhost:<port>`.
- Agents should treat the `<RUNTIME_SERVICES>` block as authoritative: don't hardcode `localhost:8000` for "the automation server", and don't probe random ports trying to discover services. If the block says automation is not running, skip `/api/automation` calls; otherwise use the listed `url_from_agent` + `api_prefix` (default `/api/automation`) and the `X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY` header.
- The launcher → backend → frontend → suffix plumbing is:
  - `scripts/runtime-services-info.mjs::buildRuntimeServicesInfo()` — dependency-free module that constructs the info object; also runs as a CLI for the Docker entrypoint. Re-exported by `scripts/dev-safe.mjs` for backward compat.
  - `scripts/dev-with-automation.mjs::buildAutomationRuntimeServicesInfo()` — wraps it with automation details. `dev-with-automation`, `dev-static`, and the published binary pass the JSON to `scripts/ingress.mjs` or `scripts/static-server.mjs` via `--runtime-services-info`.
  - `scripts/ingress.mjs` and `scripts/static-server.mjs` proxy the real agent-server `/server_info` response and append `runtime_services` when configured. This keeps version/tool compatibility fields authoritative from the SDK while letting the Agent Canvas stack advertise automation/frontend/ingress topology.
  - `src/api/agent-server-adapter.ts::fetchBackendRuntimeServicesInfo()` reads `runtime_services` from cached or freshly fetched `/server_info`; `buildRuntimeServicesSystemSuffix()` renders the `<RUNTIME_SERVICES>` markdown block; `buildAgentContext()` attaches it to `agent_context.system_message_suffix` when present.
  - E2E coverage: the mock-LLM automation test (`tests/e2e/mock-llm/automations/mock-llm-automation.spec.ts`) verifies the `<RUNTIME_SERVICES>` block reaches the LLM via `getMockLLMRequests()` and checks for Agent Server, Automation backend, and `/api/automation` entries.

### `/server_info.runtime_services` shape

The `runtime_services` value is a JSON object of:

```json
{
  "mode": "dev:automation",
  "services": {
    "agent_server": {
      "description": "The OpenHands Agent Server this agent is running inside. ...",
      "url_from_agent": "http://localhost:18000"
    },
    "ingress": {
      "description": "Unified entry point. Routes /api/automation/* ...",
      "url_from_agent": "http://localhost:8000"
    },
    "frontend": {
      "kind": "vite",
      "description": "Vite dev server hosting the agent-canvas frontend.",
      "url_from_agent": "http://localhost:3001"
    },
    "automation": {
      "description": "OpenHands Automations service. All routes are mounted under '/api/automation'. Authenticate with header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'.",
      "url_from_agent": "http://localhost:18001",
      "api_prefix": "/api/automation",
      "docs_url": "http://localhost:18001/api/automation/docs",
      "openapi_url": "http://localhost:18001/api/automation/openapi.json",
      "auth_env_var": "OPENHANDS_AUTOMATION_API_KEY"
    }
  }
}
```

All keys under `services` are optional and omitted when the corresponding service isn't running. `frontend.kind` is `"vite"` for dev launchers running the Vite dev server and `"static"` for stacks serving a pre-built `build/` directory (`dev:static`, the published `agent-canvas` binary).

### Example `<RUNTIME_SERVICES>` block (dev with automation)

```
<RUNTIME_SERVICES>
You are running inside an agent-canvas dev stack started in 'dev:automation' mode.
The following services are reachable from your sandbox. URLs are written
from your point of view (i.e., as you should curl/fetch them).

* Agent Server (you): http://localhost:18000
    The OpenHands Agent Server this agent is running inside. Tool calls (terminal, file_editor, browser, etc.) execute here.
* Ingress: http://localhost:8000
    Unified entry point. Routes /api/automation/* to the automation backend, /api/* and /sockets to the agent-server, and /* to the frontend.
* Frontend: http://localhost:3001
    Vite dev server hosting the agent-canvas frontend.
* Automation backend: http://localhost:18001
    OpenHands Automations service. All routes are mounted under '/api/automation'. Authenticate with header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'.
    Docs:    http://localhost:18001/api/automation/docs
    OpenAPI: http://localhost:18001/api/automation/openapi.json
    Auth:    header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'

Trust this block over guessing: do not assume any other URLs are running.
In particular, http://localhost:18000 inside your sandbox is the Agent Server
you are running inside of — NOT the automation backend.
</RUNTIME_SERVICES>
```


- README expectation: keep the first section as a concrete, chronological from-scratch quickstart for running this frontend against a real `openhands-agent-server` (clone, install prerequisites, optional `.env`, run `npm run dev`).
- Windows-specific command syntax (PowerShell) lives in `README.windows.md`. When changing install / Docker sandbox instructions in `README.md`, update `README.windows.md` in the same PR to keep them in sync.
- `scripts/dev-safe.mjs` uses `uvx` for temporary agent-server installation — no permanent `uv tool install` needed. Environment variables (highest precedence first):
  - `OH_AGENT_SERVER_LOCAL_PATH` — absolute path to a local `software-agent-sdk` checkout. Runs the local checkout via `uvx` with `--with-editable` for `openhands-sdk`/`openhands-tools`/`openhands-workspace` and `--reinstall` for `openhands-agent-server`, so SDK edits are picked up on restart. Highest precedence.
  - `OH_AGENT_SERVER_GIT_REF` — git commit SHA or branch name (takes precedence over version)
  - `OH_AGENT_SERVER_VERSION` — specific PyPI version (e.g., "1.49.5")
  - `OH_SECRET_KEY` — secret key for settings encryption; auto-generated and persisted to `~/.openhands/agent-canvas/secret-key.txt` on first run (same file Docker uses), ensuring dev mode and Docker share the same key when both mount the same `~/.openhands` directory. Override with the env var to pin a specific key.
  - `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0` / `VITE_SESSION_API_KEY` — session API key for agent-server authentication; auto-generated using `crypto.randomBytes(32)` if not set, passed to both agent-server (`OH_SESSION_API_KEYS_0`) and frontend (`VITE_SESSION_API_KEY`)
  - Default: released PyPI version `1.49.5` for agent-server SDK libraries

- Security: launchers generate and persist a 64-character session API key at `~/.openhands/agent-canvas/session-api-key.txt` unless overridden. The agent-server and automation backend share that session key. `OH_SECRET_KEY` protects settings encryption and is persisted separately at `~/.openhands/agent-canvas/secret-key.txt`.
- `scripts/dev-safe.mjs` should fail fast if `uvx` cannot be spawned (for example missing PATH entries).
- `tools/` holds Python modules the agent-server can import: `buildAgentServerEnv` exposes the directory through `OH_EXTRA_PYTHON_PATH` (Docker: `/opt/agent-canvas/tools`, see `docker/entrypoint.sh`). `tools/canvas_ui_tool.py` is imported at startup via `--import-modules canvas_ui_tool` (appended by `buildAgentServerCommand` and by both launch lines in `docker/entrypoint.sh`), not only lazily from persisted conversation metadata: besides the legacy `canvas_ui` registration it registers the SDK's builtin `FinishTool`, so `openhands-automation` ≥ 1.9.0 presets — which dispatch remote conversations with `finish_tool_response_schema=TaskOutcome` and advertise the tool as the non-self-registering `openhands.sdk.tool.builtins.finish` — do not fail every run with `ToolDefinition 'FinishTool' is not registered`. Remove that registration once the SDK registers builtins for remote conversations.
- `npm run dev` runs the full local stack via `uvx` (agent-server + automation backend + Vite dev server + ingress proxy) with no Docker dependency. `npm run dev:static` does the same but serves a production build of the frontend instead of the Vite dev server.
- `scripts/dev-with-automation.mjs` runs the full stack: agent-server, automation backend (both via uvx), frontend server, and ingress proxy. It defaults to Vite when run directly, supports `--static` for an existing build, and supports `--dynamic` so wrappers that default static can opt back into Vite. Uses a standalone ingress proxy (`scripts/ingress.mjs`) to route traffic:
  - Keep `SIGINT`, `SIGTERM`, and `SIGHUP` wired through the coordinated shutdown handler. Services run in detached process groups on POSIX, so cleanup must use `signalProcessTree()` rather than signaling only the direct child; regression coverage lives in `__tests__/scripts/dev-with-automation.test.ts`.
  - `/api/automation/*` → automation backend (:18001)
  - `/api/*`, `/sockets`, etc. → agent server (:18000)
  - `/*` (default) → frontend server (:3001), either Vite or static depending on launcher mode
  - Environment variables: `PORT` (ingress port, default from `config/defaults.json`), `OH_AUTOMATION_GIT_REF` (git ref, overrides default version), and `OH_AUTOMATION_VERSION` (defaults to `versions.automation` in `config/defaults.json`)
  - `scripts/check-sdk-version-sync.mjs` checks the released `openhands-automation` package against `versions.agentServer` in `config/defaults.json`; these must always match — if the automation package's SDK dependencies differ from `agentServer`, the check fails.
  - Access points: `http://localhost:8000/` (main UI), `http://localhost:8000/api/automation/docs` (API docs)
  - Security: the automation backend receives the same session key as the agent-server through `AUTOMATION_LOCAL_API_KEY`; the frontend does not bake a separate automation API key.
- `scripts/ingress.mjs` is a standalone HTTP reverse proxy that can be used independently to route traffic to multiple backends based on URL path prefix.
- `scripts/dev-safe.mjs` (now `npm run dev:minimal`) runs just agent-server + Vite without automation.

- **Centralized config**: `config/defaults.json` is the single source of truth for version pins (agent-server, automation, automation SDK), port defaults, persistence paths, and package names. All consumers read from this file:
  - JS scripts (`dev-safe.mjs`, `dev-with-automation.mjs`, `check-sdk-version-sync.mjs`) read it via `JSON.parse(readFileSync(...))`.
  - Docker: a `config-gen` build stage converts the JSON to `/opt/agent-canvas/defaults.env` (shell-sourceable); `entrypoint.sh` sources it at startup.
  - CI workflow: a `Read defaults from config/defaults.json` step uses `node -p` to extract values into `$GITHUB_OUTPUT`.
  - Dockerfile ARG defaults are kept as fallbacks for local `docker build` without the CI workflow; CI always passes `--build-arg` overrides from the JSON.
  - To bump a version, edit `config/defaults.json` only — the JS scripts, Docker build, and CI workflow all derive their values from it.
- Docker all-in-one image: `.github/workflows/docker.yml` builds and publishes `ghcr.io/openhands/agent-canvas` — a combined image that bundles the agent-server (from `ghcr.io/openhands/agent-server`), the automation server (`openhands-automation` via pip), and the agent-canvas frontend (static build). The Dockerfile lives at `docker/Dockerfile`, the entrypoint at `docker/entrypoint.sh`. The workflow structure mirrors the SDK repo's `server.yml`: a `build-and-push-image` matrix job (2 × arch: amd64 on `ubuntu-24.04`, arm64 on `ubuntu-24.04-arm`) pushes arch-suffixed tags, then `merge-manifests` creates multi-arch manifests via `docker buildx imagetools create`, then `consolidate-build-info` aggregates artifacts, and `update-pr-description` updates the PR body (using `<!-- AGENT_CANVAS_DOCKER_START -->` / `<!-- AGENT_CANVAS_DOCKER_END -->` markers). The workflow triggers on push to main, `v*` tags (releases), PRs, and `workflow_dispatch`. On release tags it also pushes semver tags (e.g. `1.2.3`, `1.2`, `1`, `latest`). Fork PRs are skipped (no GHCR auth). On PRs that link an `OpenHands/software-agent-sdk` PR in the description, the Docker workflow uses that SDK PR's published branch image (`ghcr.io/openhands/agent-server:<branch-with-slashes-as-dashes>-python`) as the agent-server base image unless a `workflow_dispatch` input explicitly overrides it. The image exposes port 8000 as a unified entry point: `/api/automation/*` → automation (:18001), `/api/*` → agent-server (:18000), `/*` → static frontend. The Dockerfile accepts the public `VITE_POSTHOG_API_KEY` build arg; CI passes staging for PR/main images and production for tagged releases. The npm release workflow passes the same production key to both the app and library builds. The entrypoint auto-generates **both** the session API key and `OH_SECRET_KEY` (persisted to `~/.openhands/agent-canvas/session-api-key.txt` and `secret-key.txt` respectively) when none is provided, so the image runs secure by default. Users can override either via env var (`OH_SECRET_KEY`, `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0`). `scripts/dev-safe.mjs` uses the same `secret-key.txt` file, so dev mode and Docker share the same key when both use the same `~/.openhands` directory.

- Spec files live under `specs/`. Spec IDs are stable — never renumber. Mark deprecated specs with ~~strikethrough~~. Tag implementation code and tests with `// @spec BM-002 — Short title` comments so specs are grep-able across the codebase (`grep -rn '@spec BM-' src/ __tests__/`). Place the comment on the line immediately above the relevant code block or test. When multiple tests cover the same spec, use `it.each` if the test structure is identical.
