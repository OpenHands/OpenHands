# Enterprise Sandbox Production Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Enterprise-compatible sandbox image and a separated local control-plane/sandbox Docker topology without changing product behavior.

**Architecture:** Refactor the existing Dockerfile into named targets built from one pinned Agent Server base. Enterprise and local sandbox targets preserve the upstream entrypoint; the control-plane target reuses the local orchestration script in explicit external-Agent-Server mode, while the all-in-one target remains the compatibility default.

**Tech Stack:** Docker BuildKit, Docker Compose, Bash, Node.js ESM, Vitest, GitHub Actions, OpenHands Agent Server and Automation.

---

## File map

- `docker/Dockerfile`: shared runtime, sandbox, control-plane, and legacy targets.
- `docker/entrypoint.sh`: local control-plane and legacy orchestration only.
- `docker/compose.yml`: separated local topology and directional URLs.
- `scripts/docker-build.mjs`: deterministic target/image/platform selection.
- `scripts/docker-dev.mjs`: one-command Compose startup and secret injection.
- `config/defaults.json`: image/version/port source of truth.
- `__tests__/scripts/docker-image-contract.test.ts`: Docker and workflow boundaries.
- `__tests__/scripts/docker-build.test.ts`: build-helper behavior.
- `__tests__/scripts/docker-separated-topology.test.ts`: entrypoint and Compose wiring.
- `.github/workflows/docker.yml`: Enterprise amd64 build/publication.
- `docs/SELF_HOSTING.md`, `AGENTS.md`: operator and repository contracts.

### Task 1: Establish Docker target contracts

**Files:**
- Create: `__tests__/scripts/docker-image-contract.test.ts`
- Modify: `docker/Dockerfile`

- [ ] **Step 1: Write the failing target test**

Read `docker/Dockerfile`, parse `FROM ... AS ...` stages, and assert:

```ts
expect(stages).toEqual(expect.arrayContaining([
  "agent-runtime-base", "enterprise-sandbox", "dev-sandbox",
  "control-plane-runtime", "control-plane", "final",
]));
expect(stageParent("enterprise-sandbox")).toBe("agent-runtime-base");
expect(stageParent("dev-sandbox")).toBe("agent-runtime-base");
expect(stageBody("enterprise-sandbox"))
  .not.toMatch(/ENTRYPOINT|CMD|frontend|automation|entrypoint\.sh/i);
expect(lastStage()).toBe("final");
expect(stageBody("final")).toContain("ENTRYPOINT");
```

- [ ] **Step 2: Verify RED**

Run `npx vitest run __tests__/scripts/docker-image-contract.test.ts`.
Expected: FAIL because the targets do not exist.

- [ ] **Step 3: Add the minimal build graph**

Reshape the Dockerfile around:

```dockerfile
FROM ${AGENT_SERVER_IMAGE} AS agent-runtime-base
ARG AGENT_CANVAS_VERSION=dev
ARG AGENT_SERVER_VERSION=unknown
ARG OPENHANDS_BUILD_GIT_SHA=unknown
ARG OPENHANDS_BUILD_GIT_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/OpenHands/OpenHands" \
      org.opencontainers.image.version="${AGENT_CANVAS_VERSION}" \
      org.opencontainers.image.revision="${OPENHANDS_BUILD_GIT_SHA}" \
      dev.openhands.agent-server.version="${AGENT_SERVER_VERSION}"

FROM agent-runtime-base AS enterprise-sandbox
LABEL org.opencontainers.image.title="agent-canvas-enterprise-sandbox"

FROM agent-runtime-base AS dev-sandbox
LABEL org.opencontainers.image.title="agent-canvas-dev-sandbox"

FROM agent-runtime-base AS control-plane-runtime
# Existing Automation/frontend/defaults installation and copies.

FROM control-plane-runtime AS control-plane
ENV AGENT_CANVAS_AGENT_SERVER_MODE=external
ENTRYPOINT ["tini", "--", "/opt/agent-canvas/entrypoint.sh"]

FROM control-plane-runtime AS final
ENV AGENT_CANVAS_AGENT_SERVER_MODE=embedded
ENTRYPOINT ["tini", "--", "/opt/agent-canvas/entrypoint.sh"]
```

Do not add `ENTRYPOINT` or `CMD` to the shared or sandbox targets. Keep `final`
last so unqualified builds retain existing behavior.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused test, then commit:

```bash
git add docker/Dockerfile __tests__/scripts/docker-image-contract.test.ts
git commit -m "build: add enterprise sandbox docker target"
```

### Task 2: Make local Docker builds target-aware

**Files:**
- Create: `__tests__/scripts/docker-build.test.ts`
- Modify: `scripts/docker-build.mjs`
- Modify: `config/defaults.json`
- Modify: `package.json`

- [ ] **Step 1: Write failing helper tests**

Require import-safe `parseArgs()` and `buildDockerCommand()` helpers:

```ts
expect(parseArgs([]).target).toBe("final");
expect(parseArgs(["--enterprise"]).target).toBe("enterprise-sandbox");
expect(parseArgs(["--enterprise"]).platform).toBe("linux/amd64");
expect(parseArgs(["--enterprise"]).tag).toBe(
  `agent-canvas-enterprise-sandbox:${canvasVersion}-agent-server-${agentServerVersion}`,
);
expect(buildDockerCommand(parseArgs(["--enterprise"])))
  .toEqual(expect.arrayContaining(["--target", "enterprise-sandbox", "--platform", "linux/amd64"]));
```

Cover `--agent-server-image`, `--tag`, `--platform`, and passthrough arguments.
Reject unknown targets and non-amd64 Enterprise builds.

- [ ] **Step 2: Verify RED**

Run `npx vitest run __tests__/scripts/docker-build.test.ts`.
Expected: FAIL because the script is not import-safe or target-aware.

- [ ] **Step 3: Implement deterministic build selection**

Every command must include the selected target/platform and these build args:

```js
`AGENT_SERVER_IMAGE=${options.agentServerImage}`
`AGENT_SERVER_VERSION=${options.agentServerVersion}`
`AUTOMATION_VERSION=${config.versions.automation}`
`AGENT_CANVAS_VERSION=${packageJson.version}`
```

Add `images.enterpriseSandbox` to `config/defaults.json`, an ESM main guard, and:

```json
"build:docker:enterprise": "node scripts/docker-build.mjs --enterprise",
"build:docker:control-plane": "node scripts/docker-build.mjs --target control-plane"
```

- [ ] **Step 4: Verify GREEN and commit**

Run both Docker contract tests, then commit the four changed files with
`build: add target-aware docker commands`.

### Task 3: Support an external sandbox service

**Files:**
- Create: `__tests__/scripts/docker-separated-topology.test.ts`
- Modify: `docker/entrypoint.sh`
- Modify: `__tests__/scripts/docker-vscode-route-sync.test.ts`

- [ ] **Step 1: Write failing executable topology tests**

Extract a marked `# >>> agent-server-topology` shell block and assert:

```ts
expect(resolve({ AGENT_CANVAS_AGENT_SERVER_MODE: "embedded" }).proxyUrl)
  .toBe("http://127.0.0.1:18000");
expect(resolve({
  AGENT_CANVAS_AGENT_SERVER_MODE: "external",
  AGENT_SERVER_URL: "http://agent-server:8000",
  VSCODE_HOST: "agent-server",
}).proxyUrl).toBe("http://agent-server:8000");
expect(resolve({ AGENT_CANVAS_AGENT_SERVER_MODE: "external" }).status)
  .not.toBe(0);
```

Also assert that external mode cannot start `openhands-agent-server`, proxy
routes use `AGENT_SERVER_PROXY_URL`, and the editor route uses `VSCODE_HOST`.

- [ ] **Step 2: Verify RED**

Run `npx vitest run __tests__/scripts/docker-separated-topology.test.ts`.
Expected: FAIL because no topology mode exists.

- [ ] **Step 3: Implement explicit topology resolution**

Resolve exactly once:

```bash
AGENT_CANVAS_AGENT_SERVER_MODE="${AGENT_CANVAS_AGENT_SERVER_MODE:-embedded}"
case "$AGENT_CANVAS_AGENT_SERVER_MODE" in
  embedded)
    AGENT_SERVER_PROXY_URL="${AGENT_SERVER_URL:-http://127.0.0.1:${AGENT_SERVER_PORT}}"
    VSCODE_HOST="${VSCODE_HOST:-127.0.0.1}"
    ;;
  external)
    : "${AGENT_SERVER_URL:?AGENT_SERVER_URL is required in external mode}"
    AGENT_SERVER_PROXY_URL="$AGENT_SERVER_URL"
    VSCODE_HOST="${VSCODE_HOST:-agent-server}"
    ;;
  *) log_error "Unsupported AGENT_CANVAS_AGENT_SERVER_MODE"; exit 1 ;;
esac
```

Embedded mode preserves current secret generation and Agent Server startup.
External mode skips both, waits on the explicit health URL with a bounded Node
`fetch()` loop, and defaults `AUTOMATION_AGENT_SERVER_URL` to the proxy URL.
All proxy routes use the explicit URL. Runtime metadata uses
`SANDBOX_AGENT_SERVER_URL` for the agent self-view and `AUTOMATION_BASE_URL` for
the sandbox-to-control-plane view.

- [ ] **Step 4: Verify GREEN and commit**

Run the new topology test plus `docker-vscode-route-sync.test.ts`, then commit
with `feat: support an external sandbox service`.

### Task 4: Add the separated one-command stack

**Files:**
- Create: `docker/compose.yml`
- Create: `scripts/docker-dev.mjs`
- Modify: `__tests__/scripts/docker-separated-topology.test.ts`
- Modify: `package.json`

- [ ] **Step 1: Extend tests and confirm RED**

Require Compose services/targets and explicit directional URLs:

```ts
expect(compose).toContain("target: control-plane");
expect(compose).toContain("target: dev-sandbox");
expect(compose).toContain("AGENT_SERVER_URL: http://agent-server:8000");
expect(compose).toContain("AUTOMATION_BASE_URL: http://control-plane:8000");
expect(compose).toContain("SANDBOX_AGENT_SERVER_URL: http://localhost:8000");
```

Require `buildComposeCommand()` to inject non-empty session/encryption secrets
through the child environment, never command arguments or committed defaults.
Run the topology test and observe the expected missing-file/export failure.

- [ ] **Step 2: Implement Compose and launcher**

Compose builds `control-plane` and `dev-sandbox`, uses one private network,
publishes only the control-plane port, mounts project/state volumes, requires
`${LOCAL_BACKEND_API_KEY:?}` and `${OH_SECRET_KEY:?}`, and waits for sandbox
health. The import-safe launcher uses `randomBytes(32).toString("hex")` for
missing values and invokes:

```js
["docker", "compose", "-f", "docker/compose.yml", "up", "--build", ...args]
```

Add `"dev:docker:separated": "node scripts/docker-dev.mjs"`.

- [ ] **Step 3: Verify GREEN and commit**

Run:

```bash
npx vitest run __tests__/scripts/docker-separated-topology.test.ts
env LOCAL_BACKEND_API_KEY=test-session-key OH_SECRET_KEY=test-secret-key docker compose -f docker/compose.yml config --quiet
```

Commit with `feat: add separated docker development stack`.

### Task 5: Publish and document the Enterprise image

**Files:**
- Modify: `.github/workflows/docker.yml`
- Modify: `docs/SELF_HOSTING.md`
- Modify: `AGENTS.md`
- Modify: `__tests__/scripts/docker-image-contract.test.ts`

- [ ] **Step 1: Add failing workflow/documentation assertions**

```ts
expect(workflow).toContain("target: enterprise-sandbox");
expect(workflow).toContain("platforms: linux/amd64");
expect(workflow).toContain("agent-canvas-enterprise-sandbox");
expect(selfHosting).toContain("OpenHands Enterprise sandbox image");
expect(selfHosting).toContain("Sandbox Image Tag");
```

Run the contract test and confirm these assertions fail.

- [ ] **Step 2: Add the Enterprise workflow job**

Add an `enterprise_image` workflow input and an amd64 job that uses
`docker/build-push-action` with `target: enterprise-sandbox`, explicit Agent
Server/Canvas versions, SHA/PR tags, and stable
`<canvas-version>-agent-server-<agent-server-version>` tags. Never publish an
Enterprise `latest` tag. Inspect the result against the upstream effective
entrypoint and command.

- [ ] **Step 3: Document operation and repository contracts**

Document `npm run build:docker:enterprise`, `npm run dev:docker:separated`,
amd64, exact Enterprise version matching, registry push, Admin Console Sandbox
Image fields, secrets, and the retained legacy path. Update `AGENTS.md` with the
new Docker topology source of truth.

- [ ] **Step 4: Verify GREEN and commit**

Run all three focused Docker tests and commit with
`ci: publish enterprise sandbox images`.

### Task 6: Complete verification

**Files:** Modify only when a verification failure identifies a defect.

- [ ] **Step 1: Run focused tests**

```bash
npx vitest run __tests__/scripts/docker-image-contract.test.ts __tests__/scripts/docker-build.test.ts __tests__/scripts/docker-separated-topology.test.ts __tests__/scripts/docker-vscode-route-sync.test.ts
```

- [ ] **Step 2: Build and inspect the amd64 Enterprise image**

```bash
npm run build:docker:enterprise -- --load
docker image inspect ghcr.io/openhands/agent-server:1.44.1-python --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
docker image inspect agent-canvas-enterprise-sandbox:1.16.0-agent-server-1.44.1 --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}} {{.Architecture}}'
```

Expected: upstream and custom entrypoint/command match exactly; architecture is
`amd64`.

- [ ] **Step 3: Smoke-test the separated stack**

```bash
npm run dev:docker:separated -- --detach
curl --fail http://127.0.0.1:8000/alive
docker compose -f docker/compose.yml ps
docker compose -f docker/compose.yml down
```

Expected: both services are healthy and shutdown succeeds.

- [ ] **Step 4: Run repository gates**

```bash
npm run lint
npm test
npm run build
npm run build:lib
```

- [ ] **Step 5: Audit the branch delta**

```bash
git diff upstream/main...HEAD --check
git status --short
git log --oneline upstream/main..HEAD
```

Expected: no whitespace errors, untracked implementation artifacts, or
unplanned changes.
