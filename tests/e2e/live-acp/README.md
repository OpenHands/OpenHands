# Live ACP-in-Docker e2e

Proves the containerized ACP credential path **through Canvas's own code** —
each credential is saved to the agent-server's secret store (as onboarding does)
and `buildStartConversationRequest` references it as a `LookupSecret` the server
resolves at spawn time — against a real agent-server container, with real
provider API calls. It's the "it actually works" companion to the unit tests in
`__tests__/api/agent-server-adapter.test.ts` — those assert the request shape;
this asserts a real agent reply.

> **Use Canvas's supported Agent Server version** from `config/defaults.json`
> (currently 1.50.1; minimum 1.47.0). OpenCode first enters the server's
> `ACPServerKind` in 1.45.0. The historical 1.25.0 result below proves only the
> off-event-loop LookupSecret fix, not today's Canvas/OpenCode compatibility.

It is **not** part of `npm test` (it lives under `tests/`, which Vitest excludes,
and needs a running container + real host credentials).

## Run it

```bash
# 1. Agent-server container. Match the current Canvas release's server pin.
#    The Python mount keeps pre-migration conversation state loadable.
docker run -d --name oh-acp -p 8010:8000 \
  -v oh-acp-data:/workspace \
  -v "$(pwd)/tools:/canvas-tools:ro" -e OH_EXTRA_PYTHON_PATH=/canvas-tools \
  ghcr.io/openhands/agent-server:1.50.1-python

# 2. Run the e2e (all providers, or a subset).
npx vite-node -c tests/e2e/live-acp/vite-node.config.mts \
  tests/e2e/live-acp/acp-docker-e2e.mts -- codex claude gemini opencode

# 3. Tear down (holds real creds).
docker rm -f oh-acp
```

Credentials are read from the host and **never printed**: Codex `~/.codex/auth.json`,
the Claude Code OAuth token from the macOS keychain, the gcloud ADC for Gemini
Vertex (`gcloud auth application-default login` first), and the optional
`OPENCODE_API_KEY`. OpenCode's default `opencode/big-pickle` model can complete
the smoke test without a key; other providers are skipped when their required
host credentials are absent.

The provider plans (models, credential collectors) and the HTTP/poll helpers are
shared between both scripts via `harness.mts` — change a model default or
credential knob there, not per script.

## Last validated result (agent-server `1.25.0-python`, unified LookupSecret path)

Re-validated 2026-06-07 against `ghcr.io/openhands/agent-server:1.25.0-python`
(the first release with software-agent-sdk#3510), on a **fresh volume** — every
credential seeded from the secret store, no leftover state. Each credential
rides as a loopback `LookupSecret`; the logs confirm the agent-server resolved
it (`GET /api/settings/secrets/<name>` 200) during ACP cold-start — **no
deadlock, no "Failed to start ACP server: timed out"**, which is exactly what
#3510 fixes. Codex and Claude also passed the app-orchestrator script
(`acp-docker-app-e2e.mts`).

OpenCode was validated separately on 2026-09-28 through the app-orchestrator
script against Agent Server 1.49.6 and OpenCode 1.18.23. The real turn reached
`finished` and returned `ACPOK-OPENCODE`; the selected preset, registry command,
and `opencode/big-pickle` model also survived a browser reload.

| Provider        | Result                                      | Evidence (agent-server logs)                                                                                                                                                                                                 |
| --------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Codex**       | ✅ real reply `ACPOK-CODEX` (both scripts)  | `Materialised ACP file-secret 'CODEX_AUTH_JSON' -> …/acp/codex/auth.json`; codex-acp 0.15.0; `Authenticating with ACP method: chatgpt`                                                                                       |
| **Claude Code** | ✅ real reply `ACPOK-CLAUDE` (both scripts) | claude-agent-acp 0.30.0; `CLAUDE_CODE_OAUTH_TOKEN` env path (no `ANTHROPIC_BASE_URL`)                                                                                                                                        |
| **Gemini CLI**  | ✅ real reply `ACPOK-GEMINI`¹               | `Materialised ACP file-secret 'GOOGLE_APPLICATION_CREDENTIALS_JSON' -> …/acp/gemini-cli/gcloud-credentials.json`; gemini-cli 0.45.1; `Authenticating with ACP method: vertex-ai` → real Vertex inference on `gemini-2.5-pro` |
| **OpenCode**    | ✅ real reply `ACPOK-OPENCODE`              | OpenCode 1.18.23 starts through the shared registry command; default `opencode/big-pickle`; optional `OPENCODE_API_KEY` is emitted as a `LookupSecret` when present                                                          |

¹ **Gemini prerequisites.** The full turn passes with: a **fresh** host ADC
(`gcloud auth application-default login` — a stale one fails as `invalid_rapt`,
a credential problem, not a Canvas one), the **non-flash** `gemini-2.5-pro`
model (gemini-cli 0.45.x re-resolves any `*-flash` id at generation time to its
current default flash, which 404s on projects that don't serve it —
software-agent-sdk#3532; this is why Canvas preselects `gemini-2.5-pro`), and
`ACP_E2E_GEMINI_SESSION_MODE=default` to clear the separate gemini-cli ≥0.43
`set_session_mode("yolo")` headless-init blocker.

## Knobs

- `ACP_E2E_BASE_URL` (default `http://localhost:8010`)
- `ACP_E2E_SESSION_API_KEY` (optional `X-Session-API-Key` for an authenticated
  Agent Server; never printed)
- `ACP_E2E_WORKING_DIR_BASE` (default `/workspace/acp-e2e` for the
  request-builder script and `/workspace/app-e2e` for the app-path script)
- `ACP_E2E_CODEX_MODEL` / `ACP_E2E_CLAUDE_MODEL` / `ACP_E2E_GEMINI_MODEL` /
  `ACP_E2E_OPENCODE_MODEL`
- `ACP_E2E_EXPECTED_REPLY` (optional human-readable reply override; defaults to
  the provider's short smoke-test token)
- `ACP_E2E_GEMINI_SESSION_MODE` (set `default` to bypass the SDK `yolo` blocker)
- `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` (else read from gcloud / `us-central1`)
