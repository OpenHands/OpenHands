/**
 * Mock-LLM E2E tests for cross-connection between frontend-only and
 * backend-only deployments.
 *
 * Verifies that a frontend-only instance can connect to one or more
 * separate backend-only instances through the browser UI:
 *
 *   1. **Single backend**: A frontend-only instance connects to a
 *      standalone backend-only instance. The user adds the backend
 *      through the Manage Backends modal and verifies the app loads.
 *
 *   2. **Multiple backends**: A frontend-only instance connects to
 *      two separate backend-only instances and switches between them.
 *
 * These tests spawn their own child processes (not the webServer
 * entries in playwright.mock-llm.config.ts) so each test controls
 * exactly which partial-stack flags and ports are used.
 */

import { test, expect, type Page } from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// ── Paths ──────────────────────────────────────────────────────────────
const PROJECT_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../..",
);
const BIN = join(PROJECT_ROOT, "bin/agent-canvas.mjs");

// ── Port ranges (high ports unlikely to collide with other tests) ──────
//
// Each instance needs a base port; createIsolatedEnv allocates +1, +2, +3
// for sub-services.
const CROSS_FE_PORT = "18370";
const CROSS_BE_A_PORT = "18380";
const CROSS_BE_B_PORT = "18390";

// ── Helpers ────────────────────────────────────────────────────────────

function spawnAgentCanvas(
  flags: string[],
  env: Record<string, string> = {},
): ChildProcess {
  return spawn("node", [BIN, ...flags], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      VITE_DO_NOT_TRACK: "1",
      VITE_ENABLE_BROWSER_TOOLS: "false",
      ...env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function collectOutput(child: ChildProcess): { get(): string } {
  let buf = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    buf += chunk.toString();
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    buf += chunk.toString();
  });
  return { get: () => buf };
}

function createIsolatedEnv(
  ingressPort: string,
  extra: Record<string, string> = {},
): { env: Record<string, string>; stateDir: string; sessionKey: string } {
  const stateDir = mkdtempSync(join(tmpdir(), "cross-connect-"));
  const sessionKey = randomBytes(32).toString("hex");
  return {
    stateDir,
    sessionKey,
    env: {
      OH_CANVAS_SAFE_STATE_DIR: stateDir,
      PORT: ingressPort,
      LOCAL_BACKEND_API_KEY: sessionKey,
      OH_CANVAS_SAFE_BACKEND_PORT: String(parseInt(ingressPort) + 1),
      OH_CANVAS_SAFE_AUTOMATION_PORT: String(parseInt(ingressPort) + 2),
      OH_CANVAS_SAFE_VITE_PORT: String(parseInt(ingressPort) + 3),
      OH_SESSION_API_KEY_PATH: join(stateDir, "session-key.txt"),
      ...extra,
    },
  };
}

async function killChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve();
    }, 5_000);
    child.on("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

/**
 * Poll a URL until it returns a non-5xx response or timeout.
 * Returns the HTTP status or null if the service never became ready.
 */
async function pollUrl(
  url: string,
  timeoutMs = 120_000,
  intervalMs = 1_000,
): Promise<number | null> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(url, {
        signal: AbortSignal.timeout(2_000),
      });
      if (resp.status < 500) return resp.status;
    } catch {
      // Connection refused — not ready yet
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return null;
}

/** Dismiss the analytics consent modal if it appears. */
async function dismissAnalyticsModal(page: Page) {
  try {
    const confirmButton = page.getByRole("button", {
      name: "Confirm preferences",
    });
    await confirmButton.click({ timeout: 3_000 });
  } catch {
    // Modal didn't appear — fine
  }
}

/** Suppress analytics modals via localStorage. */
async function suppressAnalytics(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem("analytics-consent", "false");
    window.localStorage.setItem("openhands-telemetry-consent", "denied");
    window.localStorage.setItem("openhands-telemetry-first-use", "true");
  });
}

async function addBackendViaOnboarding(
  page: Page,
  opts: { name: string; host: string; apiKey: string },
) {
  await expect(page.getByTestId("onboarding-step-check-backend")).toBeVisible({
    timeout: 15_000,
  });

  const nameInput = page.getByTestId("onboarding-backend-name");
  await nameInput.click();
  await nameInput.fill(opts.name);

  const hostInput = page.getByTestId("onboarding-backend-host");
  await hostInput.click();
  await hostInput.fill(opts.host);

  const keyInput = page.getByTestId("onboarding-backend-api-key");
  await keyInput.click();
  await keyInput.fill(opts.apiKey);

  await page.getByTestId("onboarding-backend-next").click();

  await expect(page.getByTestId("onboarding-step-choose-agent")).toBeVisible({
    timeout: 15_000,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test.describe.configure({ mode: "serial" });

// ───────────────────────────────────────────────────────────────────────
// 1. Frontend-only connects to a single backend-only instance
// ───────────────────────────────────────────────────────────────────────

test.describe("cross-connect: frontend-only → backend-only", () => {
  const children: ChildProcess[] = [];
  const stateDirs: string[] = [];

  test.afterEach(async () => {
    await Promise.all(children.map(killChild));
    children.length = 0;
    for (const dir of stateDirs) rmSync(dir, { recursive: true, force: true });
    stateDirs.length = 0;
  });

  test("frontend-only connects to a separate backend-only instance", async ({
    page,
  }) => {
    // Backend-only needs uvx → agent-server: 3+ minutes
    test.setTimeout(300_000);

    // These tests spawn bin/agent-canvas.mjs locally, which needs a
    // pre-built frontend. Skip when running in Docker-only CI (no local build).
    test.skip(
      !existsSync(join(PROJECT_ROOT, "build/index.html")),
      "build/index.html missing — skipped (run `npm run build:app` or use the npm e2e config)",
    );

    // ── 1. Start backend-only instance ────────────────────────────────
    const beEnv = createIsolatedEnv(CROSS_BE_A_PORT);
    stateDirs.push(beEnv.stateDir);
    const beOutput = collectOutput(
      (() => {
        const child = spawnAgentCanvas(["--backend-only"], beEnv.env);
        children.push(child);
        return child;
      })(),
    );

    // ── 2. Start frontend-only instance ───────────────────────────────
    const feEnv = createIsolatedEnv(CROSS_FE_PORT);
    stateDirs.push(feEnv.stateDir);
    const feOutput = collectOutput(
      (() => {
        const child = spawnAgentCanvas(["--frontend-only"], feEnv.env);
        children.push(child);
        return child;
      })(),
    );

    // ── 3. Wait for both to become ready (in parallel) ────────────────
    const feUrl = `http://localhost:${CROSS_FE_PORT}`;
    const beUrl = `http://localhost:${CROSS_BE_A_PORT}`;

    const [feStatus, beStatus] = await Promise.all([
      pollUrl(`${feUrl}/`, 60_000),
      pollUrl(`${beUrl}/server_info`, 180_000),
    ]);

    expect(
      feStatus,
      `Frontend-only never ready.\nOutput: ${feOutput.get().slice(-500)}`,
    ).toBe(200);
    expect(
      beStatus,
      `Backend-only never ready.\nOutput: ${beOutput.get().slice(-800)}`,
    ).not.toBeNull();

    // ── 4. Open browser to the frontend-only instance ─────────────────
    await suppressAnalytics(page);
    await page.goto(feUrl, { waitUntil: "domcontentloaded" });

    // First-run onboarding owns the initial backend collection before
    // the Manage Backends recovery screen is allowed to appear.
    await expect(page.getByTestId("manage-backends-modal")).not.toBeVisible({
      timeout: 1_000,
    });

    // ── 5. Add the backend-only instance via onboarding ───────────────
    await addBackendViaOnboarding(page, {
      name: "Remote Backend",
      host: beUrl,
      apiKey: beEnv.sessionKey,
    });
    await page.getByTestId("onboarding-skip").click();

    // ── 6. Reload to pick up the new backend ──────────────────────────
    // After adding a backend while the root first-run gate is active,
    // reload once to re-evaluate the root bootstrap against the newly
    // persisted active backend.
    await page.reload({ waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // ── 7. Verify the app loads ───────────────────────────────────────
    // The app should reach either the onboarding flow or the home page
    // — crucially NOT the manage-backends modal or auth screen.
    // Both can be visible at the same time (onboarding overlays the home
    // launcher), so check each individually instead of using .or() which
    // fails Playwright strict mode when both match.
    const homeVisible = await page
      .getByTestId("home-chat-launcher")
      .isVisible({ timeout: 20_000 })
      .catch(() => false);
    const onboardingVisible = await page
      .getByTestId("onboarding-step-choose-agent")
      .isVisible({ timeout: 2_000 })
      .catch(() => false);
    expect(
      homeVisible || onboardingVisible,
      "Expected either home-chat-launcher or onboarding to be visible",
    ).toBe(true);

    // The manage-backends modal and auth screen must NOT be showing.
    await expect(
      page.getByTestId("manage-backends-modal"),
    ).not.toBeVisible({ timeout: 2_000 });
    await expect(
      page.getByTestId("api-key-entry-screen"),
    ).not.toBeVisible({ timeout: 2_000 });
  });
});

// ───────────────────────────────────────────────────────────────────────
// 2. Frontend-only connects to multiple backend-only instances
// ───────────────────────────────────────────────────────────────────────

test.describe("cross-connect: frontend-only → multiple backends", () => {
  const children: ChildProcess[] = [];
  const stateDirs: string[] = [];

  test.afterEach(async () => {
    await Promise.all(children.map(killChild));
    children.length = 0;
    for (const dir of stateDirs) rmSync(dir, { recursive: true, force: true });
    stateDirs.length = 0;
  });

  test("connects to two separate backends and switches between them", async ({
    page,
  }) => {
    // Two backend-only instances via uvx — very slow
    test.setTimeout(360_000);

    test.skip(
      !existsSync(join(PROJECT_ROOT, "build/index.html")),
      "build/index.html missing — skipped (run `npm run build:app` or use the npm e2e config)",
    );

    // ── 1. Spawn all three instances concurrently ─────────────────────
    const beEnvA = createIsolatedEnv(CROSS_BE_A_PORT);
    const beEnvB = createIsolatedEnv(CROSS_BE_B_PORT);
    const feEnv = createIsolatedEnv(CROSS_FE_PORT);
    stateDirs.push(beEnvA.stateDir, beEnvB.stateDir, feEnv.stateDir);

    const beOutputA = collectOutput(
      (() => {
        const child = spawnAgentCanvas(["--backend-only"], beEnvA.env);
        children.push(child);
        return child;
      })(),
    );
    const beOutputB = collectOutput(
      (() => {
        const child = spawnAgentCanvas(["--backend-only"], beEnvB.env);
        children.push(child);
        return child;
      })(),
    );
    const feOutput = collectOutput(
      (() => {
        const child = spawnAgentCanvas(["--frontend-only"], feEnv.env);
        children.push(child);
        return child;
      })(),
    );

    // ── 2. Wait for all three to become ready ─────────────────────────
    const feUrl = `http://localhost:${CROSS_FE_PORT}`;
    const beUrlA = `http://localhost:${CROSS_BE_A_PORT}`;
    const beUrlB = `http://localhost:${CROSS_BE_B_PORT}`;

    const [feStatus, beStatusA, beStatusB] = await Promise.all([
      pollUrl(`${feUrl}/`, 60_000),
      pollUrl(`${beUrlA}/server_info`, 180_000),
      pollUrl(`${beUrlB}/server_info`, 180_000),
    ]);

    expect(
      feStatus,
      `Frontend-only never ready.\nOutput: ${feOutput.get().slice(-500)}`,
    ).toBe(200);
    expect(
      beStatusA,
      `Backend A never ready.\nOutput: ${beOutputA.get().slice(-800)}`,
    ).not.toBeNull();
    expect(
      beStatusB,
      `Backend B never ready.\nOutput: ${beOutputB.get().slice(-800)}`,
    ).not.toBeNull();

    // ── 3. Verify both backends are independently reachable ───────────
    const [infoA, infoB] = await Promise.all([
      fetch(`${beUrlA}/server_info`).then((r) => r.json()),
      fetch(`${beUrlB}/server_info`).then((r) => r.json()),
    ]);
    expect(infoA).toHaveProperty("version");
    expect(infoB).toHaveProperty("version");

    // ── 4. Open browser to the frontend-only instance ─────────────────
    await suppressAnalytics(page);
    await page.goto(feUrl, { waitUntil: "domcontentloaded" });

    // The first-run onboarding backend step should appear before the
    // Manage Backends recovery modal.
    await expect(page.getByTestId("manage-backends-modal")).not.toBeVisible({
      timeout: 1_000,
    });

    // ── 5. Add Backend A through onboarding ──────────────────────────
    await addBackendViaOnboarding(page, {
      name: "Backend A",
      host: beUrlA,
      apiKey: beEnvA.sessionKey,
    });

    await page.getByTestId("onboarding-skip").click();
    await page.reload({ waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await expect(page.getByTestId("home-chat-launcher")).toBeVisible({
      timeout: 20_000,
    });

    // ── 6. Add Backend B via the dropdown menu ────────────────────────
    // Open the backend selector dropdown in the sidebar/footer
    await page.getByTestId("backend-selector").click();
    await page.getByTestId("add-backend-menu-item").click();

    // The add-backend modal should appear
    await expect(page.getByTestId("add-backend-modal")).toBeVisible({
      timeout: 5_000,
    });

    // Fill in Backend B details
    const nameInput = page.getByTestId("add-backend-name");
    await nameInput.click();
    await nameInput.fill("Backend B");

    const hostInput = page.getByTestId("add-backend-host");
    await hostInput.click();
    await hostInput.fill(beUrlB);

    const keyInput = page.getByTestId("add-backend-api-key");
    await keyInput.click();
    await keyInput.fill(beEnvB.sessionKey);

    await page.getByTestId("add-backend-submit").click();

    // Wait for the modal to close (connection validated + saved)
    await expect(page.getByTestId("add-backend-modal")).not.toBeVisible({
      timeout: 15_000,
    });

    // ── 7. Switch to Backend B via the dropdown ───────────────────────
    // The dropdown should now list both backends. Click Backend B.
    await page.getByTestId("backend-selector").click();

    // Find "Backend B" in the dropdown options and click it
    const backendBOption = page.locator(
      '[data-testid="backend-selector"] + [role="listbox"] [role="option"]',
    );
    // If the dropdown renders options differently, fall back to text match
    const optionB = page.getByRole("option", { name: /Backend B/i });
    if (await optionB.isVisible({ timeout: 2_000 }).catch(() => false)) {
      await optionB.click();
    } else {
      // The dropdown may use a different structure — try the last option
      const options = backendBOption;
      const count = await options.count();
      if (count > 0) {
        // Click the option containing "Backend B"
        for (let i = 0; i < count; i++) {
          const text = await options.nth(i).textContent();
          if (text?.includes("Backend B")) {
            await options.nth(i).click();
            break;
          }
        }
      }
    }

    // ── 8. Verify the app works after switching ───────────────────────
    // The page should reload or re-settle. Wait for the home UI to be
    // visible, confirming Backend B is reachable.
    await page.waitForLoadState("domcontentloaded", { timeout: 15_000 });
    await dismissAnalyticsModal(page);
    await expect(page.getByTestId("home-chat-launcher")).toBeVisible({
      timeout: 20_000,
    });

    // ── 9. Verify both backends appear in the manage modal ────────────
    await page.getByTestId("backend-selector").click();
    await page.getByTestId("manage-backends-menu-item").click();
    await expect(page.getByTestId("manage-backends-modal")).toBeVisible({
      timeout: 5_000,
    });

    // Both backends should be listed
    const list = page.getByTestId("manage-backends-list");
    await expect(list.getByText("Backend A")).toBeVisible({ timeout: 5_000 });
    await expect(list.getByText("Backend B")).toBeVisible({ timeout: 5_000 });
  });
});
