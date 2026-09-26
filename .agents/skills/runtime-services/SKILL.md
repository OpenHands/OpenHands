---
name: runtime-services
description: >
  Local dev stacks and runtime services — the runtime_services metadata
  block, dev launchers (dev, dev:static, dev:minimal, dev-with-automation),
  the ingress proxy, service URLs, and example blocks. Load before working on
  local stack orchestration or dev tooling.
triggers:
  - runtime services
  - runtime_services
  - dev stack
  - dev-with-automation
  - dev-safe
  - ingress
  - local stack
---

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

- `scripts/dev-safe.mjs` uses `uvx` for temporary agent-server installation — no permanent `uv tool install` needed. Environment variables (highest precedence first):
  - `OH_AGENT_SERVER_LOCAL_PATH` — absolute path to a local `software-agent-sdk` checkout. Runs the local checkout via `uvx` with `--with-editable` for `openhands-sdk`/`openhands-tools`/`openhands-workspace` and `--reinstall` for `openhands-agent-server`, so SDK edits are picked up on restart. Highest precedence.
  - `OH_AGENT_SERVER_GIT_REF` — git commit SHA or branch name (takes precedence over version)
  - `OH_AGENT_SERVER_VERSION` — specific PyPI version (e.g., "1.49.4")
  - `OH_SECRET_KEY` — secret key for settings encryption; auto-generated and persisted to `~/.openhands/agent-canvas/secret-key.txt` on first run (same file Docker uses), ensuring dev mode and Docker share the same key when both mount the same `~/.openhands` directory. Override with the env var to pin a specific key.
  - `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0` / `VITE_SESSION_API_KEY` — session API key for agent-server authentication; auto-generated using `crypto.randomBytes(32)` if not set, passed to both agent-server (`OH_SESSION_API_KEYS_0`) and frontend (`VITE_SESSION_API_KEY`)
  - Default: released PyPI version `1.49.4` for agent-server SDK libraries
