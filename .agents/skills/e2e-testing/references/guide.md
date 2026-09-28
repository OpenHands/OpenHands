## Live End-to-End Test Framework

- The live QA path is intentionally separate from ordinary mocked Playwright coverage. If ordinary browser tests are added, keep them outside `tests/e2e/live/` so `playwright.config.ts` can run them while ignoring `**/live/**`; live LLM-backed tests must never run as part of `npm run test:e2e`.
- Live tests live under `tests/e2e/live/` and are run only through `npm run test:e2e:live`, which uses `playwright.live.config.ts`. Keep the spec names descriptive; the primary conversation smoke test is `tests/e2e/live/real-agent-server-conversation.spec.ts`.
- `npm run test:e2e:live` loads `.env` through Node's `--env-file-if-exists` flag and invokes `tests/e2e/live/scripts/run-live-e2e.mjs`. The runner validates the required local environment, explains missing credentials/prerequisites, and then runs `playwright test --config=playwright.live.config.ts`. Use `npm run test:e2e:live -- --check` to validate local setup without running the test, and pass Playwright flags after `--` (for example `npm run test:e2e:live -- --headed`).
- Local live E2E requires one LLM credential: `LIVE_E2E_LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLM_API_KEY`. Optional overrides are `LIVE_E2E_LLM_BASE_URL`, `LIVE_E2E_LLM_MODEL`, `LIVE_E2E_SESSION_API_KEY`, `LIVE_E2E_BACKEND_URL`, and `LIVE_E2E_FRONTEND_PORT`. The local runner prints which variables are missing without printing secret values.
- Live-test-only helpers belong under `tests/e2e/live/utils/`. The current helper module is `tests/e2e/live/utils/agent-server-conversation.ts`; do not put live-only helpers in the shared `tests/e2e/support/` directory.
- `playwright.live.config.ts` starts the real local Agent Server/UI stack via `npm run dev:minimal`, not MSW mocks. It uses `LIVE_E2E_SESSION_API_KEY` when set, otherwise generates a per-run random session key and passes it through `SESSION_API_KEY`, `OH_SESSION_API_KEYS_0`, and `VITE_SESSION_API_KEY`; specs that need direct backend requests must inject `X-Session-API-Key` only for the configured backend origin through `routeBackendSessionApiKey(page)`, never through global Playwright `extraHTTPHeaders`. Live tests default to frontend port `3101` and Agent Server `http://127.0.0.1:18100` so they do not accidentally reuse a normal local dev stack.
- `tests/e2e/live/utils/agent-server-conversation.ts` configures the running Agent Server before each live conversation by PATCHing `${LIVE_E2E_BACKEND_URL ?? "http://127.0.0.1:18100"}/api/settings` with LLM settings and low-risk conversation settings. LLM credentials are read from `LIVE_E2E_LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLM_API_KEY`; CI defaults use `LIVE_E2E_LLM_BASE_URL` (default `https://llm-proxy.app.all-hands.dev`) and `LIVE_E2E_LLM_MODEL` (default `openhands/claude-haiku-4-5-20251001`).
- The live conversation test should stay cheap and as deterministic as possible while still exercising one real tool call: it asks the model to run the exact `EXPECTED_BASH_COMMAND`, waits for the bash output token to appear outside the user's message in the UI, confirms a successful `ExecuteBashObservation`/`TerminalObservation` through the real Agent Server events API, and then waits for the final `EXPECTED_REPLY_TOKEN`. This exercises the real UI, Agent Server settings API, conversation creation, websocket/event path, terminal tool execution, and LLM response path. Because LLM behavior is not perfectly deterministic even at temperature 0, CI keeps one retry for live E2E; future live tests should document any expected variance and avoid prompts that require unnecessary formatting obedience.
- Live E2E must not pollute analytics. `playwright.live.config.ts` starts the app with `VITE_DO_NOT_TRACK=1`; the live helper seeds local storage with telemetry/analytics opt-out values before app code runs; and each live spec should install `guardAgainstPostHogRequests(page)` before navigation so any attempted request to `*.posthog.com` or `z.openhands.dev` is blocked locally and fails the test.
- Live Playwright videos are intentionally recorded for CI QA debugging when `LIVE_E2E_RECORD_VIDEO=on` is set; local default video mode is `retain-on-failure`. Do not add live tests that render API keys, tokens, secret values, or credential-bearing error messages in the browser. Screenshots should target a safe app/chat region such as `data-testid="chat-interface"` instead of `page.screenshot({ fullPage: true })`, and should apply `getLiveArtifactMask(page)` for text/field redaction; if a future live test must exercise sensitive UI, change that test/media path to redact the sensitive output or retain video only on failure.
- `.github/workflows/ci.yml` runs live E2E automatically after pushes reach `main`, not on pull-request events. Manual `workflow_dispatch` with a required `pr_number` remains available for pre-merge QA. Main runs test the trusted pushed commit and publish their report and media only as the workflow summary and GitHub Actions artifact. Manual PR runs retain the PR comment/media flow and must skip fork PRs before checking out PR code so LLM credentials and artifact-push tokens are never exposed to untrusted code.
- Keep live E2E secrets out of job-level `env`. The workflow should check whether credentials exist before checkout, but inject the LLM key only into the trusted step that actually runs the live test.
- The live job uploads the Playwright HTML report plus screenshot/video output as a GitHub Actions artifact, and also extracts the primary screenshot/video attachments. It converts the WebM recording to a GIF preview with `ffmpeg` so GitHub PR comments can inline the preview. Keep Playwright trace capture disabled for live tests because the setup flow sends LLM credentials to the Agent Server settings API, and traces can record request bodies. Failure messages around live Agent Server settings must not print response bodies from credential-bearing requests.
- Inline PR-comment media is stored as PR-only files under `.pr/live-e2e/<github_run_id>/` on the PR branch, not on a long-lived orphan media branch. The comment uses `raw.githubusercontent.com/<repo>/<artifact_commit>/.pr/live-e2e/...` URLs for the GIF and PNG so GitHub can render them inline. The WebM is linked as the full recording because GitHub comments do not reliably inline WebM.
- `.github/workflows/pr-artifacts.yml` owns cleanup for `.pr/live-e2e/`: it comments when `.pr/` artifacts exist, removes them after PR approval for same-repo PRs, and opens or updates a cleanup PR against `main` if artifacts reach `main` through a fork PR or a missed approval cleanup.
- The live reporting scripts live beside the live tests under `tests/e2e/live/scripts/`: `run-live-e2e.mjs`, `extract-live-e2e-media.mjs`, `render-live-e2e-report.mjs`, and `upsert-pr-comment.mjs`. Keep report/comment/local-runner logic there rather than in top-level `scripts/`, because these scripts are part of the live E2E framework.
- When changing any part of this framework — live workflow triggers, artifact publishing, `.pr` cleanup, live Playwright config, live test file layout, helper locations, local runner behavior, or report/comment scripts — update this `AGENTS.md` section in the same PR so future agents have the current operating model.

## Mock-LLM E2E Test Framework

- Mock-LLM tests live under `tests/e2e/mock-llm/` and exercise the complete stack — from the browser through the real agent-server to a scripted mock LLM server — without any real LLM credentials. Run locally with `npm run test:e2e:mock-llm`.
- **Production-fidelity launch**: The Playwright config (`playwright.mock-llm.config.ts`) starts the full `agent-canvas` stack via `bin/agent-canvas.mjs` — the same binary that `npx @openhands/agent-canvas` executes when users install the npm package. This means mock-LLM tests exercise the actual production path: pre-built static frontend + static-server.mjs + agent-server via uvx + automation backend via uvx + ingress proxy, all behind a single port.
- A pre-built `build/` directory is required. The Playwright webServer command runs `npm run build:app` when `build/index.html` is absent, but CI should run the build step explicitly for caching (`npm run build:app` in `.github/workflows/mock-llm-e2e.yml`).
- **Single ingress URL**: Tests use one URL for both the browser (`baseURL`) and backend API assertions (`BACKEND_URL`). The ingress proxy routes `/api/*` to the agent-server, `/api/automation/*` to the automation backend, and `/*` to the static frontend. Default ingress port for tests is `18300` (override via `MOCK_LLM_INGRESS_PORT` env var).
- **State isolation**: `OH_CANVAS_SAFE_STATE_DIR=.tmp/mock-llm-state` isolates test state from the user's real `~/.openhands/agent-canvas/` directory. Both `STATE_DIR` (`.tmp/mock-llm-state`) and the automation DB dir (`.tmp/automation/`) are cleaned before each test run — the automation DB now lives outside STATE_DIR at `dirname(STATE_DIR)/automation/automations.db`, mirroring Docker's `~/.openhands/automation/automations.db`.
- **Session API key**: A random key is generated per test run and passed to the stack via `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0` / `VITE_SESSION_API_KEY`. The static server injects it into `index.html` at serve time so the frontend authenticates automatically.
- **Mock LLM server** (`tests/e2e/mock-llm/scripts/mock-llm-server.py`): Python HTTP server using openhands-sdk's `TestLLM` to return scripted tool-call + text trajectories. Supports admin API endpoints for dynamic trajectory management:
  - `POST /admin/reset` — reset to the default trajectory (terminal printf + text reply); also clears the stored completion-request history
  - `POST /admin/trajectory/register` — register a named trajectory (JSON body: `{name, turns}` where each turn is `{tool_call: {name, arguments}}` or `{text: "..."}`)
  - `POST /admin/trajectory/activate` — activate a previously registered trajectory
  - `GET /admin/requests` — return the list of all `/v1/chat/completions` request bodies captured since the last reset (used by the image-upload test to verify the image was forwarded to the LLM)
  - Profile pre-flight ping: agent-server ≥ 1.43 sends a 1-token `ping` completion whenever an LLM profile is saved (`POST /api/profiles/{name}/validate`, 30 s budget in the canvas). The mock answers it with a canned `pong` instead of feeding it to `TestLLM`, so it neither consumes a scripted turn nor 500s-and-retries past the canvas timeout when the trajectory is exhausted; it is also left out of the `/admin/requests` history.
- **Real automation backend**: The automation test uses the production automation backend (started by `bin/agent-canvas.mjs`), NOT a mock server. Terminal `curl` commands from the agent hit the automation API through the ingress proxy at the test's `BACKEND_URL` (default `http://localhost:18300`). Auth uses the `X-Session-API-Key` header matching the stack's session key.
- **Test helpers** (`tests/e2e/mock-llm/utils/mock-llm-helpers.ts`): Exports `registerTrajectory()`, `activateTrajectory()`, `resetMockLLM()`, `ensureMockLLMProfile()`, `getMockLLMRequests()` (fetches captured completion bodies from `GET /admin/requests`), `IMAGE_REPLY_TOKEN` + `MINIMAL_PNG_BASE64` (constants for the image-upload spec), ACP helpers (`configureAcpAgent()`, `verifyAcpAgentSettings()`, `resetToOpenHandsAgent()`, `ACP_REPLY_TOKEN`, `MOCK_ACP_SERVER_PATH`), and more.
- **Padding response for internal LLM call**: The agent-server makes an internal LLM call (condenser/skill-analysis) before the agent's main loop starts when skills are activated. This consumes one trajectory response. Automation tests prepend a throwaway `{ text: "" }` response as padding. The conversation test does NOT need this because its user message doesn't trigger skill activation. Since `openhands-automation==1.10.0` (openhands/automation#405), the automation lifecycle spec scripts the required `finish` tool turn in its run-conversation budget — that release requires preset automation conversations to callthe `finish` tool before the run reaches COMPLETED; blank turns alone would loop and exhaust the trajectory. The mock server logs "Mock LLM exhausted after N calls" pinpoint the exact count if it drifts again.
- **Mock ACP server** (`tests/e2e/mock-llm/scripts/mock-acp-server.py`): A minimal stdio-based ACP agent that speaks JSON-RPC using the `acp` Python library (installed as a dependency of `openhands-sdk`). Handles `initialize`, `session/new`, and `session/prompt`; sends a scripted `session/update` notification with `ACP_REPLY_TOKEN` in a text content block, then returns `stop_reason: "end_turn"`. The agent-server spawns it as a subprocess via `acp_command`. Accepts `--reply-token TOKEN` to customize the reply token.
- **Test directory layout**: Specs are organized into feature subdirectories that mirror the source code structure, enabling selective test execution based on which source files changed:
  - `settings/` — LLM profile management, ACP agent config, model switching (`mock-llm-acp-agent.spec.ts`, `mock-llm-profile-management.spec.ts`, `mock-llm-model-switch.spec.ts`)
  - `conversations/` — Core conversation flow, image upload (`mock-llm-conversation.spec.ts`, `mock-llm-image-upload.spec.ts`)
  - `files/` — Files tab, Browser tab, and git control bar coverage (`mock-llm-files-and-git.spec.ts`)
  - `automations/` — Automation lifecycle, preset cards (`mock-llm-automation.spec.ts`, `mock-llm-preset-automation.spec.ts`)
  - `onboarding/` — First-run onboarding flow (`mock-llm-onboarding-happy-path.spec.ts`, `mock-llm-onboarding-regressions.spec.ts`)
  - `backends/` — Auth modes, cross-connect, partial stack (`mock-llm-auth-modes.spec.ts`, `mock-llm-cross-connect.spec.ts`, `mock-llm-partial-stack.spec.ts`)
  - `home/` — Workspace selection, folder browser (`mock-llm-folder-workspace.spec.ts`)
  - `mcp/` — MCP marketplace/server management and credential verification (`mock-llm-mcp-github.spec.ts`, `mock-llm-mcp-slack-credentials.spec.ts`)
  - `skills/` — Skill loading and activation (`mock-llm-skills.spec.ts`)
  - `canvas-extensions/` — Canvas Extension install → enable → page render → disable → uninstall lifecycle (`mock-llm-canvas-extensions.spec.ts`). The pinned agent-server predates `/api/canvas-extensions`, so the spec serves that contract from `src/fixtures/canvas-extensions/demo-page` via `page.route()`; delete the stub once the pin ships the endpoints and install the fixture by absolute path instead.
  - `regressions/` — CSS isolation, event pagination, workspace persistence (`mock-llm-ui-regressions.spec.ts`). Always included in selective runs.
- **Selective test resolver**: `tests/e2e/mock-llm/test-mapping.json` maps source paths to test subdirectories, and `tests/e2e/mock-llm/scripts/resolve-affected-tests.mjs` can resolve a changed-file list for local investigation. Post-merge CI and `workflow_dispatch` intentionally run the full suite, so this resolver is not part of the workflow path.
- Tests run serially (`workers: 1`, `mode: "serial"` per describe block). Each spec is self-contained (configures its own LLM profile, resets mock LLM in `afterEach`). The `afterEach` hook resets the mock LLM to its default trajectory so subsequent specs start fresh even when a preceding test fails.
- CI workflow: `.github/workflows/mock-llm-e2e.yml` runs the full suite after pushes reach `main` and on manual dispatch; it does not run on pull-request events. The workflow builds the frontend, starts the mock LLM server, runs the tests, uploads artifacts, and writes the rendered report to the workflow summary. Npm-path Mock-LLM CI uses `MOCK_LLM_GLOBAL_TIMEOUT_MS` (default 20 min) and the workflow deadline adds a 60-second teardown buffer; keep `playwright.mock-llm.config.ts` and `.github/workflows/mock-llm-e2e.yml` in sync if that timeout changes.
- The custom `DoneMarkerReporter` writes `.mock-llm-markers/.tests-done` after all tests complete (before webServer teardown) so the CI wrapper can detect completion and kill the lingering teardown process.

### Docker Image Testing (Shared Specs)

- The same test specs and helpers are reused to validate the Docker image via `playwright.mock-llm-docker.config.ts`. Run locally with `npm run test:e2e:mock-llm:docker` (requires Docker daemon and a built image).
- **Architecture**: The Docker config replaces the npm path's `bin/agent-canvas.mjs` webServer with a `docker run --network host` command. The mock LLM server still runs on the host. On Linux (including CI), `--network host` lets the container share the host's network stack so all `127.0.0.1` URLs work identically. On macOS/Windows Docker Desktop (bridge networking), set `MOCK_LLM_AGENT_URL=http://host.docker.internal:<port>` so the agent-server inside Docker can reach the host-side mock LLM server.
- **Dual-stack binding**: Both `scripts/static-server.mjs` and `scripts/ingress.mjs` default to `::` (dual-stack, accepting IPv4 and IPv6 connections). The Docker entrypoint passes `--host ::` explicitly. This means `localhost` is safe in both the Docker and npm Playwright configs — whether it resolves to `127.0.0.1` (IPv4) or `::1` (IPv6), the server accepts the connection. The mock LLM server URL (`MOCK_LLM_URL`) still uses `127.0.0.1` because the Python mock server is a separate process whose bind behavior we don't control.
- **Entrypoint crash resilience**: `docker/entrypoint.sh` uses a `while kill -0 "$STATIC_PID"; do sleep 10 & wait $!; done` loop instead of `wait -n "${PIDS[@]}"` (any child). If the agent-server or automation backend exits mid-test, the static-server proxy stays up and returns 502s for backend routes — the container doesn't disappear with `ECONNREFUSED`. The container exits only when the static-server (ingress) dies or on SIGTERM/SIGINT. The `sleep & wait $!` pattern ensures `wait` (a bash builtin) is the foreground op, so trapped signals fire immediately. `cleanup()` includes `exit 0` so the script terminates after a signal-triggered trap return.
- **URL split**: `mock-llm-helpers.ts` exports two mock LLM URL constants:
  - `MOCK_LLM_BASE_URL` — always `http://127.0.0.1:<port>`, used by tests for the mock LLM admin API (register/activate/reset trajectories).
  - `MOCK_LLM_AGENT_URL` — defaults to `MOCK_LLM_BASE_URL`, overridable via `MOCK_LLM_AGENT_URL` env var. Used when configuring the LLM profile (`base_url` field) — this is the URL the agent-server uses for inference calls. The npm path and Docker-with-`--network host` path use the same value; Docker on macOS needs the override.
- **Docker image**: Set `MOCK_LLM_DOCKER_IMAGE` to the image tag (default: `ghcr.io/openhands/agent-canvas:latest`). The container is started with `--rm --network host` and a unique `--name` for cleanup.
- **State isolation**: The Docker container uses its internal state directory (no host mount needed for tests). Each test run starts a fresh container.
- **Skill test volume mounts**: Tests that create files the agent-server needs to read (skill repos, user skills) require Docker volume mounts because the container has an isolated filesystem. The Docker config mounts `.tmp/mock-llm-skill-repos/` → `/tmp/mock-llm-skill-repos/` for project skills and `.tmp/mock-llm-user-skills/` → `/home/openhands/.openhands/skills/` for user skills. Env vars `MOCK_LLM_SKILL_REPOS_CONTAINER_DIR` and `MOCK_LLM_USER_SKILLS_HOST_DIR` tell `skill-test-helpers.ts` which paths to use for agent-server API registration vs. host-side file operations.
- CI workflow: `.github/workflows/mock-llm-docker-e2e.yml` has two triggers, both using an already-built image from GHCR: (1) `workflow_run` fires automatically after a successful `Docker` workflow on `main`; (2) `workflow_dispatch` accepts a custom `docker_image` input. It does not run on pull-request events. The default image tag is derived from the tested commit SHA (`ghcr.io/openhands/agent-canvas:sha-<short>-amd64`). Report artifacts go to `test-results-mock-llm-docker/` and `playwright-report-mock-llm-docker/`.

## Debugging E2E Test Failures

When an E2E test fails in CI, use this workflow to diagnose the root cause efficiently:

### 1. Read the workflow summary first
The mock-LLM E2E workflows write a structured report to the GitHub Actions workflow summary with a test results table, pass/fail status, and collapsible failure details including the Playwright error message. **Start here** — the error message usually reveals whether the failure is a locator mismatch, a timeout, or a missing element.

### 2. Download CI artifacts
Every failing test run uploads artifacts (`mock-llm-e2e-results` for npm, `mock-llm-docker-e2e-results` for Docker). Download them with:
```bash
gh run download <run_id> --repo OpenHands/OpenHands --name mock-llm-e2e-results --dir /tmp/artifacts
```
Artifacts contain:
- `test-results-mock-llm/` — per-test directories with `test-failed-N.png` (screenshot at failure) and `error-context.md` (Playwright page snapshot as YAML accessibility tree + test source with the failing line marked)
- `playwright-report-mock-llm/` — full HTML report (`npx playwright show-report /tmp/artifacts/playwright-report-mock-llm`)

### 3. Inspect the error-context.md page snapshot
The `error-context.md` file contains a YAML accessibility tree of the entire page at the moment of failure. This is the single most useful artifact — it shows exactly what DOM elements exist, which tabs are selected, what text is in inputs, and whether a component rendered at all. Search for the element your test expects (e.g. `llm-provider-input`) to see if it's present or absent, and check surrounding context (tab selection state, form view mode, etc.) to understand why.

### 4. Common failure patterns

**"element(s) not found"** — The locator matched zero elements. The component either:
- Didn't render (conditional rendering path not taken — check the page snapshot for what DID render)
- Has a different `name`/`data-testid` than expected
- Is behind a lazy-load boundary that hasn't resolved

**Stale state from earlier serial tests** — Mock-LLM tests run serially (`workers: 1`) against a real agent-server. Earlier tests (conversation, automation) persist settings on the server. If your test depends on "clean" state but a prior test configured `llm_base_url`, `llm_model`, etc., the form may render in a different view mode. Use Playwright `page.route()` to intercept and normalize the settings response. Example: `routeOnboardingLlmCatalog` in `tests/e2e/support/onboarding-helpers.ts` intercepts `GET /api/settings` to clear `llm_base_url` so the LLM form always opens in "Basic" view.

**View mode mismatch (Basic vs Advanced)** — `LlmSettingsScreen` switches between "Basic" (renders `ModelSelector` with provider/model dropdowns) and "Advanced" (renders plain text inputs). The view is determined by `getInitialView()` which checks `currentSettings.llm_base_url` — a non-default base URL triggers "Advanced" view. If your test expects `input[name="llm-provider-input"]` but sees text inputs instead, the settings have a stale `base_url`.

**Playwright route interception vs real server** — In mock-LLM tests, routes registered with `page.route()` intercept at the browser level before requests reach the real agent-server. However, `page.route()` must be set up BEFORE `page.goto()`. The `showOnboarding` helper handles this correctly (routes are registered before navigation). Non-GET methods should use `route.fallback()` to pass through to the real server.

### 5. Running locally
```bash
npm run test:e2e:mock-llm                    # full suite
npm run test:e2e:mock-llm -- --headed        # watch in browser
npm run test:e2e:mock-llm -- -g "test name"  # run single test by name
```

## Testing Rules

<TESTING_RULES>
Create TDD tests for behavioral changes. Focus on user behavior and follow TDD best practices, including:

- AAA structure (Arrange, Act, Assert)
- Clear test focus
- Proper test data management

Before writing any test:

- Avoid duplicating test cases or logic
- Do not assert the same condition more than once
- Do not mock the hook. Instead, mock the underlying service that the hook depends on
- Prefer adding to or extending existing test files whenever possible. Create new test files only if no suitable ones exist
- Avoid brittle visual-presentation assertions. Functional CSS contracts such as style scoping and selector transformation may be tested directly
- Keep the number of test cases to the minimum necessary while still fully covering the intended changes and behaviors

Ensure each test is meaningful, concise, and covers a unique aspect of user interaction.
</TESTING_RULES>
