/**
 * Mock-LLM E2E test: GitHub MCP server install via the MCP marketplace UI.
 *
 * This test exercises the full MCP install flow — navigating to the MCP page,
 * finding the GitHub marketplace card, opening the install modal, filling in
 * the PAT field, and submitting. The `POST /api/mcp/test` endpoint is
 * intercepted to return a mock success response so the test doesn't need to
 * contact GitHub's hosted MCP endpoint.
 *
 * Verifies:
 *   1. The MCP page renders with the GitHub marketplace card visible
 *   2. Clicking the add control opens the install modal with the hosted endpoint
 *   3. Filling in the PAT and submitting succeeds (with mocked test endpoint)
 *   4. After install the GitHub server appears in the installed list
 *   5. The installed server can be deleted via the UI
 */

import { test, expect, type Page } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  ensureMockLLMProfile,
} from "../utils/mock-llm-helpers";

const FAKE_PAT = "github_pat_test_1234567890abcdef";
const GITHUB_HOSTED_MCP_URL = "https://api.githubcopilot.com/mcp/";

test.describe.configure({ mode: "serial" });

async function openGitHubInstallModal(page: Page) {
  const modal = page.getByTestId("mcp-install-modal");

  await expect(async () => {
    await page.getByTestId("mcp-marketplace-toggle-github").click();
    await expect(modal).toBeVisible({ timeout: 1_500 });
  }).toPass({ timeout: 10_000 });

  return modal;
}

test.describe("MCP GitHub server install flow", () => {
  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    // Clear any MCP servers so subsequent tests start clean
    await request
      .patch(`${BACKEND_URL}/api/settings`, {
        headers: {
          "X-Session-API-Key": SESSION_API_KEY,
          "Content-Type": "application/json",
        },
        data: { agent_settings_diff: { mcp_config: null } },
      })
      .catch(() => {});
  });

  test("step 1: GitHub card is visible on the MCP marketplace page", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the marketplace section to render
    await waitForTestId(page, "mcp-marketplace-section");
    const marketplaceGrid = page.getByTestId("mcp-marketplace-grid");
    await expect(marketplaceGrid).toBeVisible({ timeout: 10_000 });

    // Verify the GitHub card exists
    const githubCard = page.getByTestId("mcp-marketplace-card-github");
    await expect(githubCard).toBeVisible();

    // The card should display the name "GitHub"
    await expect(githubCard).toContainText("GitHub");
  });

  test("step 2: clicking GitHub add control opens the install modal with correct fields", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "mcp-marketplace-grid");

    const modal = await openGitHubInstallModal(page);

    // Verify the modal is for the GitHub entry
    await expect(modal).toHaveAttribute("data-marketplace-id", "github");

    // The modal should show the hosted streamable HTTP endpoint.
    const urlField = page.getByTestId("mcp-install-field-url");
    await expect(urlField).toBeVisible();
    await expect(urlField).toHaveValue(GITHUB_HOSTED_MCP_URL);

    // The PAT field should be present and empty.
    const patField = page.getByTestId("mcp-install-field-api_key");
    await expect(patField).toBeVisible();
    await expect(patField).toHaveValue("");
  });

  test("step 3: full install flow — fill PAT, submit, verify installed", async ({
    page,
  }) => {
    // We need an LLM profile configured for settings to work properly
    await ensureMockLLMProfile(page);

    await routeSessionApiKey(page);

    // Intercept the MCP test endpoint to return success; the test environment
    // should not contact GitHub's hosted MCP endpoint. Mirror the real
    // agent-server success shape: `tools` (a required field) lists the
    // advertised tools including the `get_me` credential probe, and
    // `tool_result` carries that read-only probe's passing outcome.
    await page.route("**/api/mcp/test", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          ok: true,
          tools: ["get_me", "list_issues"],
          tool_result: { is_error: false, text: '{"login":"octocat"}' },
        }),
      });
    });

    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "mcp-marketplace-grid");

    const modal = await openGitHubInstallModal(page);

    // Fill in the PAT — SettingsInput puts data-testid on the <input> directly
    const patInput = page.getByTestId("mcp-install-field-api_key");
    await patInput.fill(FAKE_PAT);

    // Click install
    await page.getByTestId("mcp-install-submit").click();

    // The modal should close after successful install
    await expect(modal).not.toBeVisible({ timeout: 10_000 });

    // The GitHub server should now appear in the installed list
    const installedList = page.getByTestId("mcp-installed-list");
    await expect(installedList).toBeVisible({ timeout: 10_000 });

    // The installed server card should exist
    const serverItems = installedList.getByTestId("mcp-server-item");
    await expect(serverItems.first()).toBeVisible();

    // Verify via the settings API that the server was actually persisted
    const settingsResp = await page.request.get(`${BACKEND_URL}/api/settings`, {
      headers: { "X-Session-API-Key": SESSION_API_KEY },
    });
    expect(settingsResp.ok()).toBe(true);
    const settings = await settingsResp.json();
    const mcpConfig = settings?.agent_settings?.mcp_config;
    expect(mcpConfig).toBeTruthy();

    // The GitHub server should be stored as a hosted streamable HTTP server,
    // keyed by the catalog slug ("github") so it is referenceable by name in
    // mcp_server_refs — not the auto-generated "shttp" fallback. The settings
    // API redacts persisted secrets, so the raw PAT must not be readable after
    // installation.
    expect(mcpConfig?.github).toMatchObject({
      url: GITHUB_HOSTED_MCP_URL,
      auth: { strategy: "api_key", value: "**********" },
    });
  });

  test("step 4: installed GitHub server can be deleted", async ({ page }) => {
    // First install the server via the API so we have something to delete
    const installResp = await page.request.patch(
      `${BACKEND_URL}/api/settings`,
      {
        headers: {
          "X-Session-API-Key": SESSION_API_KEY,
          "Content-Type": "application/json",
        },
        data: {
          agent_settings_diff: {
            mcp_config: {
              github: {
                url: GITHUB_HOSTED_MCP_URL,
                auth: { strategy: "api_key", value: FAKE_PAT },
              },
            },
          },
        },
      },
    );
    expect(installResp.ok()).toBe(true);

    await routeSessionApiKey(page);
    await page.goto("/mcp", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // The installed list should show the github server
    const installedList = page.getByTestId("mcp-installed-list");
    await expect(installedList).toBeVisible({ timeout: 10_000 });

    const serverItem = installedList.getByTestId("mcp-server-item").first();
    await expect(serverItem).toBeVisible();

    await serverItem.click();
    await expect(page.getByTestId("mcp-custom-editor")).toBeVisible({
      timeout: 5_000,
    });
    await page.getByTestId("mcp-custom-editor-delete").click();

    // A confirmation modal should appear — click confirm
    const confirmButton = page.getByTestId("confirm-button");
    await expect(confirmButton).toBeVisible({ timeout: 5_000 });
    await confirmButton.click();

    // After deletion the installed list should show the empty state
    await expect(page.getByTestId("mcp-installed-empty")).toBeVisible({
      timeout: 10_000,
    });

    // Verify via the settings API that the server was removed
    const settingsResp = await page.request.get(`${BACKEND_URL}/api/settings`, {
      headers: { "X-Session-API-Key": SESSION_API_KEY },
    });
    expect(settingsResp.ok()).toBe(true);
    const settings = await settingsResp.json();
    const mcpConfig = settings?.agent_settings?.mcp_config;
    const githubStillPresent = mcpConfig?.github != null;
    expect(githubStillPresent).toBe(false);
  });
});
