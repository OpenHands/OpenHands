import { defineConfig, devices } from "@playwright/test";

/**
 * Bind-policy E2E: static-server session-key injection + optional live
 * ingress at http://127.0.0.1:8000 (set OH_INGRESS_URL to override).
 *
 * Does not start Vite. Chromium only.
 *
 *   npx playwright test --config=playwright.bind-policy.config.ts
 */
export default defineConfig({
  testDir: "./tests/e2e/bind-policy",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  timeout: 60_000,
  reporter: [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    ignoreHTTPSErrors: true,
    trace: "off",
    video: "off",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
