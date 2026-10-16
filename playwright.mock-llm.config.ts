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
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import {
  allocateMockLlmPorts,
  mockLlmPortSpecs,
  releasePortsCommand,
} from "./tests/e2e/mock-llm/utils/allocate-mock-llm-ports";
import { uvCacheEnvForIsolatedHome } from "./tests/e2e/mock-llm/utils/uv-cache-env";

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

// Isolated process home so the stack does not write into the developer's
// real home. User skills live at dirname(STATE_DIR)/skills.
const TEST_HOME = resolve(
  process.env.MOCK_LLM_TEST_HOME ?? join(RUN_ROOT, "home"),
);
process.env.MOCK_LLM_TEST_HOME = TEST_HOME;

// HOME is the isolated test home so the stack writes outside the developer's
// real home. User skills are loaded from OH_PERSISTENCE_DIR/skills (the parent
// of STATE_DIR). uv keeps using the runner home, where CI pre-warmed the cache.
const runnerHome = process.env.HOME ?? homedir();
const uvCacheEnv = uvCacheEnvForIsolatedHome(runnerHome, process.env);

const SKILL_REPOS_DIR = resolve(
  process.env.MOCK_LLM_SKILL_REPOS_HOST_DIR ?? join(RUN_ROOT, "skill-repos"),
);
process.env.MOCK_LLM_SKILL_REPOS_HOST_DIR = SKILL_REPOS_DIR;

const USER_SKILLS_DIR = resolve(
  process.env.MOCK_LLM_USER_SKILLS_HOST_DIR ??
    join(dirname(STATE_DIR), "skills"),
);
process.env.MOCK_LLM_USER_SKILLS_HOST_DIR = USER_SKILLS_DIR;

// Playwright re-imports this config in each worker. allocateMockLlmPorts
// records the first result in MOCK_LLM_PORT_SNAPSHOT so workers reuse it
// after the webServers have bound the ports. Every port stays held in this
// process until its webServer command or spec releases it.
const rawPorts = await allocateMockLlmPorts(mockLlmPortSpecs("npm"), "npm");

const MOCK_LLM_PORT = rawPorts.mockLlm;
const INGRESS_PORT = rawPorts.ingress;
const PUBLIC_MODE_PORT = rawPorts.publicMode;
const BACKEND_PORT = rawPorts.backend;
const AUTOMATION_PORT = rawPorts.automation;
const VITE_PORT = rawPorts.vite;

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
      command:
        releasePortsCommand(["mockLlm"]) +
        `exec ${MOCK_LLM_PYTHON} tests/e2e/mock-llm/scripts/mock-llm-server.py --port ${MOCK_LLM_PORT}`,
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
        releasePortsCommand(["ingress", "backend", "automation", "vite"]) +
        [
          "exec env",
          envAssignment("OH_CANVAS_SAFE_STATE_DIR", STATE_DIR),
          envAssignment("PORT", INGRESS_PORT),
          envAssignment("LOCAL_BACKEND_API_KEY", sessionApiKey),
          envAssignment("OH_CANVAS_SAFE_BACKEND_PORT", BACKEND_PORT),
          envAssignment("OH_CANVAS_SAFE_AUTOMATION_PORT", AUTOMATION_PORT),
          envAssignment("OH_CANVAS_SAFE_VITE_PORT", VITE_PORT),
          envAssignment("HOME", TEST_HOME),
          envAssignment("UV_CACHE_DIR", uvCacheEnv.UV_CACHE_DIR),
          envAssignment(
            "UV_PYTHON_INSTALL_DIR",
            uvCacheEnv.UV_PYTHON_INSTALL_DIR,
          ),
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
      // orphaning to PID 1 while still holding this run's ports. Asking
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
      command:
        releasePortsCommand(["publicMode"]) +
        [
          // No DigitalOcean bridge on the e2e public-mode server: nothing here
          // uses it, and the Providers entry must not appear in these specs.
          "exec env",
          envAssignment("MARS_WEB", "0"),
          "node scripts/static-server.mjs",
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
