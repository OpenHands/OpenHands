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
import { dirname, join, resolve } from "node:path";

// ── Port allocation (separate from live E2E / dev to avoid collisions) ─
const MOCK_LLM_PORT = process.env.MOCK_LLM_PORT ?? "9999";

// The agent-canvas binary exposes a single ingress port that routes:
//   /api/automation/* → automation backend
//   /api/*, /sockets  → agent-server
//   /*                → static frontend
// Tests use this single URL for both the browser (baseURL) and backend API
// calls (the ingress proxies /api/* transparently).
const INGRESS_PORT = process.env.MOCK_LLM_INGRESS_PORT ?? "18300";

// A second static-server instance for public-mode auth tests. It serves
// the same build/ directory with --auth-required (no baked session key)
// and proxies to the same backend.
const PUBLIC_MODE_PORT = process.env.MOCK_LLM_PUBLIC_MODE_PORT ?? "18301";

// ── Session API key ────────────────────────────────────────────────────
const sessionApiKey =
  process.env.MOCK_LLM_SESSION_API_KEY?.trim() ||
  randomBytes(32).toString("hex");
process.env.MOCK_LLM_SESSION_API_KEY = sessionApiKey;

// ── State directory (isolated per test run) ────────────────────────────
const STATE_DIR = resolve(".tmp/mock-llm-state");

// Automation DB lives at $parent_of_STATE_DIR/automation/automations.db,
// mirroring docker/entrypoint.sh which uses $HOME/.openhands/automation/automations.db.
// Both STATE_DIR and AUTOMATION_DB_DIR must be cleaned between runs to avoid stale data.
const AUTOMATION_DB_DIR = join(dirname(STATE_DIR), "automation");

// ── URLs ───────────────────────────────────────────────────────────────
const INGRESS_URL = `http://localhost:${INGRESS_PORT}/`;
const MOCK_LLM_URL = `http://127.0.0.1:${MOCK_LLM_PORT}`;

// Python binary for the mock server — defaults to "python3" but CI can
// point this at a venv (e.g. ".mock-llm-venv/bin/python3") to avoid
// PEP 668 "externally managed" errors on Ubuntu 24.04+.
const MOCK_LLM_PYTHON = process.env.MOCK_LLM_PYTHON ?? "python3";

// Export for the test helpers — BACKEND_URL points to the ingress (API
// calls are proxied to the agent-server, so no direct backend port needed).
process.env.MOCK_LLM_BACKEND_URL = `http://localhost:${INGRESS_PORT}`;
process.env.MOCK_LLM_PORT = MOCK_LLM_PORT;
process.env.MOCK_LLM_PUBLIC_MODE_URL = `http://localhost:${PUBLIC_MODE_PORT}`;

function shellQuote(value: string) {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function envAssignment(name: string, value: string) {
  return `${name}=${shellQuote(value)}`;
}

export default defineConfig({
  testDir: "./tests/e2e/mock-llm",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  timeout: 60_000,
  globalTimeout: process.env.CI ? 600_000 : 0, // 10 min hard cap in CI
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
      reuseExistingServer: !process.env.CI,
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
        // Clean state dir and automation DB dir to avoid stale data between runs.
        // Automation DB is stored outside STATE_DIR (at AUTOMATION_DB_DIR) so both
        // must be cleaned; see scripts/dev-with-automation.mjs startAutomationBackend.
        `node -e "const fs=require('node:fs'); fs.rmSync('${STATE_DIR}',{recursive:true,force:true}); fs.rmSync('${AUTOMATION_DB_DIR}',{recursive:true,force:true});" && ` +
        // Build frontend if not already built (CI should pre-build for caching)
        "[ -f build/index.html ] || npm run build:app && " +
        [
          "exec env",
          envAssignment("OH_CANVAS_SAFE_STATE_DIR", STATE_DIR),
          envAssignment("PORT", INGRESS_PORT),
          envAssignment("LOCAL_BACKEND_API_KEY", sessionApiKey),
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
      url: `http://localhost:${INGRESS_PORT}/api/automation/v1`,
      timeout: 180_000, // allow extra time for build + agent-server + automation startup
      reuseExistingServer: !process.env.CI,
    },
    // 3. Public-mode static server — same build/, same backend, but with
    //    --auth-required (no session key injected). The agent-server's
    //    internal ports are the defaults from config/defaults.json (18000
    //    for agent-server, 18001 for automation).
    {
      command: [
        "exec node scripts/static-server.mjs",
        "--dir build",
        `--port ${PUBLIC_MODE_PORT}`,
        "--auth-required",
        "--route /api/automation=http://localhost:18001",
        "--route /api=http://localhost:18000",
        "--route /server_info=http://localhost:18000",
        "--route /sockets=http://localhost:18000",
      ].join(" "),
      url: `http://localhost:${PUBLIC_MODE_PORT}/`,
      timeout: 15_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
});
