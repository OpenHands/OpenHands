/**
 * Playwright config for the Windows npm-install E2E job
 * (.github/workflows/windows-install-e2e.yml).
 *
 * Unlike playwright.mock-llm.config.ts, this config does not start Agent
 * Canvas. The workflow installs the packed npm tarball globally and runs the
 * `agent-canvas` command the way a Windows user does (README Option 1), then
 * points these specs at that stack. Only the mock LLM server starts here.
 *
 * Required environment:
 *   MOCK_LLM_BACKEND_URL      URL of the running Agent Canvas ingress
 *   MOCK_LLM_SESSION_API_KEY  the key the launcher persisted on first run
 *   MOCK_LLM_PYTHON           Python with openhands-sdk installed
 */

import { defineConfig, devices } from "@playwright/test";

const MOCK_LLM_PORT = process.env.MOCK_LLM_PORT ?? "9999";
const CANVAS_URL = process.env.MOCK_LLM_BACKEND_URL ?? "http://127.0.0.1:8000";
const MOCK_LLM_PYTHON = process.env.MOCK_LLM_PYTHON ?? "python";

// The shared helpers read these at import time.
process.env.MOCK_LLM_BACKEND_URL = CANVAS_URL;
process.env.MOCK_LLM_PORT = MOCK_LLM_PORT;

export default defineConfig({
  testDir: "./tests/e2e/mock-llm",
  // First-run onboarding plus a conversation that runs a terminal command:
  // the path a new Windows user takes. Separators are matched either way so
  // the patterns also work on POSIX hosts.
  testMatch: [
    /onboarding[\\/]mock-llm-onboarding-happy-path\.spec\.ts$/,
    /conversations[\\/]mock-llm-conversation\.spec\.ts$/,
    // Open Windows bugs; these fail on Windows until each issue is fixed.
    /windows[\\/]mock-llm-windows-known-issues\.spec\.ts$/,
  ],
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  // The conversation spec's "Active" badge poll races profile
  // reconciliation (see its comments) and hit 2 of 4 Windows runs. One
  // retry keeps the job usable; Playwright still reports such runs as flaky.
  retries: 1,
  workers: 1,
  // Windows runners start conversations more slowly than Linux ones.
  timeout: 120_000,
  // Stop and report before the workflow's step timeout, so a hang still
  // leaves a report and logs in the artifact.
  globalTimeout: 25 * 60_000,
  reporter: [
    ["line"],
    // Records why each expected failure failed, so a known-bug test that
    // breaks for an unrelated reason can be spotted in the artifact.
    ["json", { outputFile: "test-results-windows-install/results.json" }],
    [
      "html",
      { outputFolder: "playwright-report-windows-install", open: "never" },
    ],
  ],
  outputDir: "test-results-windows-install",
  use: {
    baseURL: `${CANVAS_URL}/`,
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: `"${MOCK_LLM_PYTHON}" tests/e2e/mock-llm/scripts/mock-llm-server.py --port ${MOCK_LLM_PORT}`,
    url: `http://127.0.0.1:${MOCK_LLM_PORT}`,
    // The openhands-sdk import is slow on a cold Windows runner.
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
    stdout: "pipe",
    stderr: "pipe",
  },
});
