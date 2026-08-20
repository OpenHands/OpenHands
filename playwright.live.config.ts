import { defineConfig, devices } from "@playwright/test";
import { randomBytes } from "node:crypto";

const configuredLiveE2ESessionApiKey =
  process.env.LIVE_E2E_SESSION_API_KEY?.trim();
const liveE2ESessionApiKey =
  configuredLiveE2ESessionApiKey || randomBytes(32).toString("hex");
process.env.LIVE_E2E_SESSION_API_KEY = liveE2ESessionApiKey;
const liveE2EFrontendPort = process.env.LIVE_E2E_FRONTEND_PORT ?? "3101";
const liveE2EBackendURL =
  process.env.LIVE_E2E_BACKEND_URL ?? "http://127.0.0.1:18100";
let liveE2EBackendPort: string;
try {
  liveE2EBackendPort = new URL(liveE2EBackendURL).port || "18100";
} catch {
  throw new Error("Invalid LIVE_E2E_BACKEND_URL. Expected an absolute URL.");
}
const liveE2EFrontendURL = `http://localhost:${liveE2EFrontendPort}/`;
const liveE2EVideoMode =
  process.env.LIVE_E2E_RECORD_VIDEO === "on" ? "on" : "retain-on-failure";

function shellQuote(value: string) {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function envAssignment(name: string, value: string) {
  return `${name}=${shellQuote(value)}`;
}

export default defineConfig({
  testDir: "./tests/e2e/live",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 180_000,
  reporter: [
    ["line"],
    ["json", { outputFile: "test-results-live/results.json" }],
    ["html", { outputFolder: "playwright-report-live", open: "never" }],
  ],
  outputDir: "test-results-live",
  use: {
    baseURL: liveE2EFrontendURL,
    screenshot: "only-on-failure",
    trace: "off",
    video: liveE2EVideoMode,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  // This live stack is intended for macOS/Linux shells and Ubuntu CI runners;
  // the command below intentionally uses POSIX environment assignment syntax.
  webServer: {
    command:
      "node -e \"const fs=require('node:fs'); for (const p of ['.tmp/live-e2e-state','node_modules/.vite']) fs.rmSync(p,{recursive:true,force:true});\" && " +
      [
        "OH_CANVAS_SAFE_STATE_DIR=.tmp/live-e2e-state",
        envAssignment("LOCAL_BACKEND_API_KEY", liveE2ESessionApiKey),
        envAssignment("OH_CANVAS_SAFE_BACKEND_PORT", liveE2EBackendPort),
        "VITE_DO_NOT_TRACK=1",
        "VITE_ENABLE_BROWSER_TOOLS=false",
        envAssignment("VITE_FRONTEND_PORT", liveE2EFrontendPort),
        "npm run dev:minimal",
      ].join(" "),
    url: liveE2EFrontendURL,
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
