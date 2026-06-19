/**
 * Mock-LLM E2E test: ACP "credentials configured" auth banner (issue #1244).
 *
 * Regression coverage for the banner state added in #1244. On a backend where
 * the host-login probe can't confirm a session — a Docker/cloud agent-server,
 * which ships the ACP wrappers but not the interactive `claude` CLI — the probe
 * classifies as `unknown`. Before the fix the banner rendered nothing, so a
 * provider credential saved to the backend store looked identical to "no
 * credentials at all". The fix surfaces a neutral "credentials configured"
 * banner from the secret store (the one signal that works on Docker/cloud)
 * WITHOUT ever claiming a verified host login.
 *
 * Flow (Settings → Agent, the same way a user configures a built-in provider):
 *   1. Switch the agent type to ACP and select the Claude Code preset.
 *   2. With no secret saved, the "configured" banner is absent.
 *   3. Save a Claude credential (CLAUDE_CODE_OAUTH_TOKEN). The value is never
 *      read or validated by the banner — the store only needs the name present —
 *      so a placeholder is sufficient and no real/PAYG credential is required.
 *   4. The neutral "configured" banner appears, and the green "signed in"
 *      banner does NOT (the honesty guard — a stored credential is not a login).
 *
 * Determinism note: the assertion requires the host-login probe to be
 * inconclusive (`unknown`). That is always true on the containerized
 * agent-server this suite runs against (no interactive `claude` CLI) and on a
 * clean CI runner. If the suite is ever run against a backend whose host HAS a
 * real Claude login, the probe legitimately reports "signed in" and the
 * "configured" precondition can't be reached — the test skips with a reason
 * rather than false-failing.
 */

import type { Page } from "@playwright/test";
import { test, expect } from "@playwright/test";
import {
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  selectDropdownOption,
  resetToOpenHandsAgentViaUI,
  BACKEND_URL,
  SESSION_API_KEY,
} from "../utils/mock-llm-helpers";

const CLAUDE_PROVIDER_NAME = "Claude Code";
const OAUTH_TOKEN_SECRET = "CLAUDE_CODE_OAUTH_TOKEN";
// A placeholder value — the banner reads only whether the secret NAME exists in
// the backend store; it never authenticates, so no real credential is needed.
const PLACEHOLDER_TOKEN = "e2e-placeholder-not-a-real-token";

/**
 * Seed onboarding flags + a single local backend, like the shared
 * {@link seedLocalStorage}, but point the backend at ``BACKEND_URL`` rather than
 * ``window.location.origin``. In CI those are the same value (the config serves
 * the browser and the backend from one ingress origin), so behaviour is
 * unchanged there; pointing at ``BACKEND_URL`` additionally lets the suite run
 * against a standalone backend (e.g. ``examples/acp-docker`` on a separate port
 * during local development).
 */
async function seedBackend(page: Page) {
  await page.addInitScript(
    ({ host, apiKey }) => {
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          { id: "default-local", name: "Local", host, apiKey, kind: "local" },
        ]),
      );
    },
    { host: BACKEND_URL, apiKey: SESSION_API_KEY },
  );
}

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM ACP credentials-configured banner (#1244)", () => {
  test.beforeAll(async ({ request }) => {
    // Start from a clean store so the no-secret baseline holds even if a prior
    // run's best-effort afterAll cleanup didn't complete (crash / network blip).
    await request
      .delete(`${BACKEND_URL}/api/settings/secrets/${OAUTH_TOKEN_SECRET}`, {
        headers: { "X-Session-API-Key": SESSION_API_KEY },
      })
      .catch(() => {
        // best-effort — the secret may simply not exist yet
      });
  });

  test.beforeEach(async ({ page }) => {
    await seedBackend(page);
  });

  test.afterAll(async ({ request, browser }) => {
    // Remove the saved credential so other suites start from a clean store.
    try {
      await request.delete(
        `${BACKEND_URL}/api/settings/secrets/${OAUTH_TOKEN_SECRET}`,
        { headers: { "X-Session-API-Key": SESSION_API_KEY } },
      );
    } catch {
      // best-effort
    }
    // Reset agent_kind back to OpenHands so suites expecting the default agent
    // aren't affected by our ACP selection.
    const page = await browser.newPage();
    try {
      await seedBackend(page);
      await resetToOpenHandsAgentViaUI(page);
    } catch {
      // best-effort
    } finally {
      await page.close();
    }
  });

  test("Claude credential in the store surfaces a 'configured' banner, never 'signed in'", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/settings/agent", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "agent-settings-screen");

    const configuredBanner = page.getByTestId("settings-acp-auth-configured");
    const signedInBanner = page.getByTestId("settings-acp-auth-detected");
    const checkingBanner = page.getByTestId("settings-acp-auth-checking");

    // Arm the probe-settled wait BEFORE selecting the provider (selecting the
    // Claude Code preset is what fires the host-login probe). The probe POSTs
    // the provider's status command to the agent-server bash endpoint, so
    // settling on that response is deterministic. Waiting for the "checking"
    // banner to hide is not: `waitFor({ state: "hidden" })` is satisfied by
    // "not in the DOM", so it resolves immediately if the spinner hasn't
    // painted yet, and the skip-precondition below would then be read too
    // early — on a host with a real login the probe later reports "signed in"
    // and the test false-fails instead of skipping (see #1244).
    const probeSettled = page
      .waitForResponse(
        (res) =>
          res.url().includes("/api/bash/execute_bash_command") &&
          res.request().method() === "POST",
        { timeout: 10_000 },
      )
      .catch(() => null);

    // ── Select ACP → Claude Code ─────────────────────────────────────────
    await selectDropdownOption(page, /Agent/, /ACP/);
    await selectDropdownOption(
      page,
      /Preset/,
      new RegExp(CLAUDE_PROVIDER_NAME),
    );

    // The Claude credential field renders for the built-in provider.
    const tokenField = page.getByTestId(
      `settings-acp-secret-${OAUTH_TOKEN_SECRET}`,
    );
    await expect(tokenField).toBeVisible({ timeout: 10_000 });

    // Let the host-login probe settle: wait for its bash response, then for
    // React Query to render the terminal banner state (the spinner clears).
    // The probe is gated to local backends and runs once per provider. If it
    // reports a real login, the "configured" precondition is unreachable on
    // this host — skip rather than false-fail.
    await probeSettled;
    await expect(checkingBanner).toHaveCount(0, { timeout: 10_000 });
    test.skip(
      (await signedInBanner.count()) > 0,
      "host-login probe reported a verified login; the credentials-configured " +
        "state only applies when the probe is inconclusive (Docker/cloud/CI)",
    );

    // ── Baseline: no secret yet → banner absent ──────────────────────────
    await expect(configuredBanner).toHaveCount(0);

    // ── Save a placeholder Claude credential ─────────────────────────────
    await tokenField.click();
    await tokenField.fill(PLACEHOLDER_TOKEN);

    const saveBtn = page.getByTestId("agent-save-button");
    await expect(saveBtn).toBeEnabled({ timeout: 5_000 });
    await saveBtn.click();
    // Save completes: isDirty flips back to false and the button disables.
    await expect(saveBtn).toBeDisabled({ timeout: 10_000 });

    // ── The neutral "configured" banner appears … ────────────────────────
    await expect(configuredBanner).toBeVisible({ timeout: 10_000 });
    await expect(configuredBanner).toContainText(/configured/i);

    // ── … and the green "signed in" banner never does (honesty guard). ───
    await expect(signedInBanner).toHaveCount(0);
  });
});
