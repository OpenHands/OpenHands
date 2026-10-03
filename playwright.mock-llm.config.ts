/**
 * Playwright config for mock-LLM E2E tests.
 *
 * Starts three processes:
 *   1. Mock LLM server (Python, using openhands-sdk TestLLM)
 *   2. Full agent-canvas stack via bin/agent-canvas.mjs (agent-server +
 *      automation backend + static frontend + ingress proxy), matching the
 *      production npm-published binary.
 *   3. A second static-server instance with `--auth-required` (public mode)
 *      on a separate port, proxying to the same backend.  Used by the
 *      auth-mode E2E tests.
 *
 * The test creates an LLM profile via the UI that points at the mock server,
 * so no real LLM credentials are needed.
 *
 * A pre-built `build/` directory is required — the Playwright webServer
 * command runs `npm run build:app` when `build/index.html` is absent.
 * CI should run the build step explicitly before the tests for caching.
 */

import { defineConfig, devices } from "@playwright/test";
import { randomBytes } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { findFreePorts } from "./scripts/dev-safe.mjs";

// ── Unique run namespace & directory isolation ─────────────────────────
const runId =
  process.env.MOCK_LLM_RUN_ID?.trim() || randomBytes(4).toString("hex");
process.env.MOCK_LLM_RUN_ID = runId;

const RUN_ROOT = resolve(
  process.env.MOCK_LLM_RUN_DIR ?? join(".tmp", `mock-llm-${runId}`),
);
process.env.MOCK_LLM_RUN_DIR = RUN_ROOT;

const STATE_DIR = resolve(
  process.env.OH_CANVAS_SAFE_STATE_DIR ?? join(RUN_ROOT, "state"),
);
process.env.OH_CANVAS_SAFE_STATE_DIR = STATE_DIR;

// Persistence dir is parent of STATE_DIR, so here it is RUN_ROOT.
// All profiles, secrets, agent-profiles live inside RUN_ROOT and stay isolated.
const AUTOMATION_DB_DIR = resolve(
  process.env.MOCK_LLM_AUTOMATION_DIR ?? join(RUN_ROOT, "automation"),
);
process.env.MOCK_LLM_AUTOMATION_DIR = AUTOMATION_DB_DIR;

// Isolated test home for user-skills and user state so developer's
// real ~/.openhands/skills is NEVER touched.
const TEST_HOME = resolve(
  process.env.MOCK_LLM_TEST_HOME ?? join(RUN_ROOT, "home"),
);
process.env.MOCK_LLM_TEST_HOME = TEST_HOME;

const SKILL_REPOS_DIR = resolve(
  process.env.MOCK_LLM_SKILL_REPOS_HOST_DIR ?? join(RUN_ROOT, "skill-repos"),
);
process.env.MOCK_LLM_SKILL_REPOS_HOST_DIR = SKILL_REPOS_DIR;

const USER_SKILLS_DIR = resolve(
  process.env.MOCK_LLM_USER_SKILLS_HOST_DIR ??
    join(TEST_HOME, ".openhands", "skills"),
);
process.env.MOCK_LLM_USER_SKILLS_HOST_DIR = USER_SKILLS_DIR;
mkdirSync(USER_SKILLS_DIR, { recursive: true });

// ── Port allocation (separate from live E2E / dev to avoid collisions) ─
const rawPorts = await findFreePorts([
  {
    name: "mockLlm",
    preferred: parseInt(process.env.MOCK_LLM_PORT ?? "9999", 10),
  },
  {
    name: "ingress",
    preferred: parseInt(
      process.env.MOCK_LLM_INGRESS_PORT ?? process.env.PORT ?? "18300",
      10,
    ),
  },
  {
    name: "publicMode",
    preferred: parseInt(process.env.MOCK_LLM_PUBLIC_MODE_PORT ?? "18301", 10),
  },
  {
    name: "backend",
    preferred: parseInt(process.env.OH_CANVAS_SAFE_BACKEND_PORT ?? "18000", 10),
  },
  {
    name: "automation",
    preferred: parseInt(
      process.env.OH_CANVAS_SAFE_AUTOMATION_PORT ?? "18001",
      10,
    ),
  },
  {
    name: "vite",
    preferred: parseInt(process.env.OH_CANVAS_SAFE_VITE_PORT ?? "3001", 10),
  },
  {
    name: "feOnly",
    preferred: parseInt(process.env.MOCK_LLM_FE_ONLY_PORT ?? "18310", 10),
  },
  {
    name: "beOnly",
    preferred: parseInt(process.env.MOCK_LLM_BE_ONLY_PORT ?? "18320", 10),
  },
  {
    name: "crossFe",
    preferred: parseInt(process.env.MOCK_LLM_CROSS_FE_PORT ?? "18370", 10),
  },
  {
    name: "crossBeA",
    preferred: parseInt(process.env.MOCK_LLM_CROSS_BE_A_PORT ?? "18380", 10),
  },
  {
    name: "crossBeB",
    preferred: parseInt(process.env.MOCK_LLM_CROSS_BE_B_PORT ?? "18390", 10),
  },
]);

const MOCK_LLM_PORT = String(rawPorts.mockLlm);
const INGRESS_PORT = String(rawPorts.ingress);
const PUBLIC_MODE_PORT = String(rawPorts.publicMode);
const BACKEND_PORT = String(rawPorts.backend);
const AUTOMATION_PORT = String(rawPorts.automation);
const VITE_PORT = String(rawPorts.vite);
const FE_ONLY_PORT = String(rawPorts.feOnly);
const BE_ONLY_PORT = String(rawPorts.beOnly);
const CROSS_FE_PORT = String(rawPorts.crossFe);
const CROSS_BE_A_PORT = String(rawPorts.crossBeA);
const CROSS_BE_B_PORT = String(rawPorts.crossBeB);

process.env.MOCK_LLM_PORT = MOCK_LLM_PORT;
process.env.MOCK_LLM_INGRESS_PORT = INGRESS_PORT;
process.env.MOCK_LLM_PUBLIC_MODE_PORT = PUBLIC_MODE_PORT;
process.env.OH_CANVAS_SAFE_BACKEND_PORT = BACKEND_PORT;
process.env.OH_CANVAS_SAFE_AUTOMATION_PORT = AUTOMATION_PORT;
process.env.OH_CANVAS_SAFE_VITE_PORT = VITE_PORT;
process.env.MOCK_LLM_FE_ONLY_PORT = FE_ONLY_PORT;
process.env.MOCK_LLM_BE_ONLY_PORT = BE_ONLY_PORT;
process.env.MOCK_LLM_CROSS_FE_PORT = CROSS_FE_PORT;
process.env.MOCK_LLM_CROSS_BE_A_PORT = CROSS_BE_A_PORT;
process.env.MOCK_LLM_CROSS_BE_B_PORT = CROSS_BE_B_PORT;

// ── Session API key ────────────────────────────────────────────────────
const sessionApiKey =
  process.env.MOCK_LLM_SESSION_API_KEY?.trim() ||
  randomBytes(32).toString("hex");
process.env.MOCK_LLM_SESSION_API_KEY = sessionApiKey;

// ── URLs ───────────────────────────────────────────────────────────────
const INGRESS_URL = `http://127.0.0.1:${INGRESS_PORT}/`;
const MOCK_LLM_URL = `http://127.0.0.1:${MOCK_LLM_PORT}`;

// Python binary for the mock server — defaults to "python3" but CI can
// point this at a venv (e.g. ".mock-llm-venv/bin/python3") to avoid
// PEP 668 "externally managed" errors on Ubuntu 24.04+.
const MOCK_LLM_PYTHON = process.env.MOCK_LLM_PYTHON ?? "python3";

// Export for the test helpers — BACKEND_URL points to the ingress (API
// calls are proxied to the agent-server, so no direct backend port needed).
process.env.MOCK_LLM_BACKEND_URL = `http://127.0.0.1:${INGRESS_PORT}`;
process.env.MOCK_LLM_PUBLIC_MODE_URL = `http://127.0.0.1:${PUBLIC_MODE_PORT}`;

function shellQuote(value: string) {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function envAssignment(name: string, value: string) {
  return `${name}=${shellQuote(value)}`;
}

const DEFAULT_CI_GLOBAL_TIMEOUT_MS = 1_200_000;
const configuredCiGlobalTimeoutMs = Number.parseInt(
  process.env.MOCK_LLM_GLOBAL_TIMEOUT_MS ??
    String(DEFAULT_CI_GLOBAL_TIMEOUT_MS),
  10,
);
const ciGlobalTimeoutMs = Number.isFinite(configuredCiGlobalTimeoutMs)
  ? configuredCiGlobalTimeoutMs
  : DEFAULT_CI_GLOBAL_TIMEOUT_MS;

export default defineConfig({
  testDir: "./tests/e2e/mock-llm",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  timeout: 60_000,
  globalTimeout: process.env.CI ? ciGlobalTimeoutMs : 0, // 20 min hard cap in CI
  globalTeardown: resolve("tests/e2e/mock-llm/utils/global-teardown.ts"),
  reporter: [
    ["line"],
    ["json", { outputFile: "test-results-mock-llm/results.json" }],
    ["html", { outputFolder: "playwright-report-mock-llm", open: "never" }],
    ["./tests/e2e/mock-llm/reporters/done-marker-reporter.ts"],
  ],
  outputDir: "test-results-mock-llm",
  use: {
    baseURL: INGRESS_URL,
    screenshot: "only-on-failure",
    trace: "on-first-retry",
    video: "on",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    // 1. Mock LLM server (Python)
    {
      command: `${MOCK_LLM_PYTHON} tests/e2e/mock-llm/scripts/mock-llm-server.py --port ${MOCK_LLM_PORT}`,
      url: MOCK_LLM_URL,
      timeout: 30_000,
      reuseExistingServer: false,
      stdout: "pipe",
      stderr: "pipe",
    },
    // 2. Full agent-canvas stack via bin/agent-canvas.mjs
    //
    // This mirrors the production `npx @openhands/agent-canvas` path:
    //   - Pre-built static frontend served via static-server.mjs
    //   - Agent-server via uvx
    //   - Automation backend via uvx
    //   - Ingress proxy unifying all routes on a single port
    //
    // `exec` replaces the shell so Playwright's tracked PID IS the node
    // process. SIGTERM goes directly to the shutdown handler, which
    // kills children via process groups and exits cleanly.
    {
      command:
        // Clean isolated run root (or state dir and automation DB dir) before run
        `node -e "const fs=require('node:fs'); fs.rmSync('${RUN_ROOT}',{recursive:true,force:true}); fs.rmSync('${STATE_DIR}',{recursive:true,force:true}); fs.rmSync('${AUTOMATION_DB_DIR}',{recursive:true,force:true});" && ` +
        // Build frontend if not already built (CI should pre-build for caching)
        "[ -f build/index.html ] || npm run build:app && " +
        [
          "exec env",
          envAssignment("OH_CANVAS_SAFE_STATE_DIR", STATE_DIR),
          envAssignment("PORT", INGRESS_PORT),
          envAssignment("LOCAL_BACKEND_API_KEY", sessionApiKey),
          envAssignment("OH_CANVAS_SAFE_BACKEND_PORT", BACKEND_PORT),
          envAssignment("OH_CANVAS_SAFE_AUTOMATION_PORT", AUTOMATION_PORT),
          envAssignment("OH_CANVAS_SAFE_VITE_PORT", VITE_PORT),
          envAssignment("HOME", TEST_HOME),
          "VITE_DO_NOT_TRACK=1",
          "VITE_ENABLE_BROWSER_TOOLS=false",
          // Bypass npm — exec directly into node so SIGTERM reaches
          // the shutdown handler (npm swallows it).
          "node --env-file-if-exists=.env bin/agent-canvas.mjs",
        ].join(" "),
      // Probe the automation list endpoint through the ingress to ensure
      // the FULL stack (agent-server + automation backend + ingress) is
      // up before tests start. The automation backend starts last via
      // uvx and can take 30-60s — checking only the ingress root or
      // /server_info would let tests begin before it's ready.
      // GET /api/automation/v1 returns 200 (empty list) without auth
      // because the dev automation backend does not enforce session-key
      // auth on the list endpoint (confirmed in CI).
      url: `http://127.0.0.1:${INGRESS_PORT}/api/automation/v1`,
      timeout: 180_000, // allow extra time for build + agent-server + automation startup
      reuseExistingServer: false,
      // Without this, Playwright tears the webServer down with
      // process.kill(-pid, "SIGKILL"), which the stack cannot catch. Its
      // services are spawned detached (see scripts/dev-process-utils.mjs), so
      // they sit in their own process groups and survive that group kill,
      // orphaning to PID 1 while still holding 18000/18001/18300/3001. Asking
      // for SIGTERM lets the existing shutdown handler in
      // scripts/dev-with-automation.mjs run, which stops each service, waits
      // 3s, then force-kills stragglers. The timeout below leaves headroom
      // over that 3s pass.
      gracefulShutdown: { signal: "SIGTERM", timeout: 15_000 },
    },
    // 3. Public-mode static server — same build/, same backend, but with
    //    --auth-required (no session key injected). Proxies to the dynamically
    //    configured agent-server and automation ports.
    {
      command: [
        "exec node scripts/static-server.mjs",
        "--dir build",
        `--port ${PUBLIC_MODE_PORT}`,
        "--host 127.0.0.1",
        "--auth-required",
        `--route /api/automation=http://localhost:${AUTOMATION_PORT}`,
        `--route /api=http://localhost:${BACKEND_PORT}`,
        `--route /server_info=http://localhost:${BACKEND_PORT}`,
        `--route /sockets=http://localhost:${BACKEND_PORT}`,
      ].join(" "),
      url: `http://127.0.0.1:${PUBLIC_MODE_PORT}/`,
      timeout: 15_000,
      reuseExistingServer: false,
    },
  ],
});
