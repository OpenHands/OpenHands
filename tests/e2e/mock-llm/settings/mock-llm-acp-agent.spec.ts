/**
 * Mock-LLM E2E test: ACP (Agent Client Protocol) agent conversation.
 *
 * Exercises the full ACP agent stack through the browser UI — the same
 * way a real user would configure and use an ACP agent:
 *
 *   1. Navigate to Settings → Agent and switch the agent type to ACP
 *   2. Select "Custom" preset, paste the mock ACP server command, save
 *   3. Reload and verify the UI reflects the saved ACP configuration
 *   4. Start a conversation from the home page and verify the ACP agent
 *      responds with the expected reply token
 *   5. Resume the conversation from the sidebar after navigating away
 *
 * The mock ACP server (`mock-acp-server.py`) speaks JSON-RPC over stdio
 * and replies with a deterministic token so the test can verify the full
 * round-trip without any real LLM.
 */

import { test, expect } from "@playwright/test";
import {
  ACP_REPLY_TOKEN,
  MOCK_ACP_COMMAND_PYTHON,
  MOCK_ACP_COMMAND_SCRIPT,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  deleteConversation,
  resetToOpenHandsAgentViaUI,
  resetMockLLM,
  ensureMockLLMProfile,
  ensureMockLLMAgentProfile,
  openAgentProfileEditor,
  selectDropdownOption,
  setChatInput,
  BACKEND_URL,
  SESSION_API_KEY,
} from "../utils/mock-llm-helpers";

const USER_MESSAGE = "Hello ACP agent, please reply.";

/**
 * The command string the user types into the ACP command textarea.
 *
 * Uses environment-aware paths: in Docker E2E the agent-server runs
 * inside a container, so we use the container-side paths set by the
 * Docker Playwright config; in the npm path we use host-local paths.
 */
const ACP_COMMAND_TEXT = `${MOCK_ACP_COMMAND_PYTHON} ${MOCK_ACP_COMMAND_SCRIPT}`;

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM ACP agent conversation", () => {
  let conversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterAll(async ({ request, browser }) => {
    // Clean up the conversation
    if (conversationId) {
      try {
        await deleteConversation(request, conversationId);
      } catch {
        // best-effort
      }
    }

    // Reset agent-server back to OpenHands via the Settings → Agent UI
    // + restore mock LLM profile so subsequent test suites (which expect
    // agent_kind=openhands) are not affected by our ACP configuration.
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await ensureMockLLMAgentProfile(page.request);
      await resetToOpenHandsAgentViaUI(page);
      await ensureMockLLMProfile(page);
    } catch {
      // best-effort
    } finally {
      await page.close();
    }
    try {
      await resetMockLLM(request);
    } catch {
      // best-effort
    }
  });

  // ── Step 1: Configure ACP agent through the Settings → Agent UI ─────

  test("step 1: configure ACP agent via Settings → Agent UI", async ({
    page,
    request,
  }) => {
    // The agent-server may make internal LLM calls (condenser) even for
    // ACP conversations. Ensure a mock LLM profile exists so those calls
    // don't fail. This UI flow is not what we're testing — the ACP UI is.
    await ensureMockLLMProfile(page);

    // Settings → Agent is the Agent Profile library (#1571); edit the
    // seeded "default" profile through it rather than a standalone form.
    await openAgentProfileEditor(page, "default");

    // ── Switch agent type from OpenHands → ACP ──

    await test.step("select ACP agent type", async () => {
      await selectDropdownOption(page, /Agent/, /ACP/);
    });

    // ── After selecting ACP, the preset dropdown + command fields appear ──

    await test.step("select Custom preset and enter command", async () => {
      // Wait for the ACP-specific fields to appear
      await waitForTestId(page, "agent-preset-selector");

      // Select "Custom" preset so we can enter our own command
      await selectDropdownOption(page, /Preset/, /Custom/);

      // Fill in the ACP command pointing to our mock server
      const commandInput = page.getByTestId("agent-command-input");
      await expect(commandInput).toBeVisible({ timeout: 5_000 });
      await commandInput.click();
      await commandInput.fill(ACP_COMMAND_TEXT);
    });

    // ── Save the configuration ──

    await test.step("save agent settings", async () => {
      const saveBtn = page.getByTestId("save-agent-profile-btn");
      await expect(saveBtn).toBeEnabled({ timeout: 5_000 });
      await saveBtn.click();

      // A successful save returns the editor to the profile list.
      await waitForTestId(page, "add-agent-profile", 10_000);
    });

    // ── Verify: agent-profiles API reflects the ACP configuration ──

    await test.step("verify agent-profiles API reflects ACP config", async () => {
      const resp = await request.get(
        `${BACKEND_URL}/api/agent-profiles/default`,
        {
          headers: { "X-Session-API-Key": SESSION_API_KEY },
        },
      );
      expect(
        resp.ok(),
        `GET /api/agent-profiles/default returned ${resp.status()}`,
      ).toBe(true);
      const detail = await resp.json();
      const profile = detail?.profile as Record<string, unknown>;
      expect(profile?.agent_kind).toBe("acp");

      // The agent-profiles API stores acp_command as a shell string
      // (ts-client `AgentProfile.acp_command: string | null`), unlike the
      // legacy `/api/settings` which exposed a token array.
      const command = profile?.acp_command;
      expect(
        typeof command === "string" && command.includes("mock-acp-server.py"),
        `acp_command should reference mock-acp-server.py, got: ${JSON.stringify(command)}`,
      ).toBe(true);
    });
  });

  // ── Step 2: Reload and verify the UI reflects saved ACP config ──────

  test("step 2: reload and verify ACP settings are persisted in UI", async ({
    page,
  }) => {
    // Settings → Agent is the Agent Profile library (#1571); re-open the
    // "default" profile's editor rather than a standalone form.
    await openAgentProfileEditor(page, "default");

    // The command textarea should show the mock server command
    const commandInput = page.getByTestId("agent-command-input");
    await expect(commandInput).toBeVisible({ timeout: 5_000 });
    const commandValue = await commandInput.inputValue();
    expect(
      commandValue.includes("mock-acp-server.py"),
      `Command should contain mock-acp-server.py after reload, got: "${commandValue}"`,
    ).toBe(true);

    // The preset dropdown should indicate the custom server is active.
    // Since we used a custom command that doesn't match any built-in
    // provider, the preset should show "Custom".
    const presetSelector = page.getByTestId("agent-preset-selector");
    await expect(presetSelector).toBeVisible({ timeout: 5_000 });
  });

  // ── Step 3: Start an ACP conversation from the home page ────────────

  test("step 3: start ACP conversation and verify agent reply", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    // Passively capture POST /api/conversations payload to verify ACP tags
    let capturedPayload: Record<string, unknown> | null = null;
    const capturePayload = (req: import("@playwright/test").Request) => {
      if (
        req.method() === "POST" &&
        new URL(req.url()).pathname === "/api/conversations"
      ) {
        try {
          capturedPayload = req.postDataJSON();
        } catch {
          // non-JSON body
        }
      }
    };
    page.on("request", capturePayload);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the home page chat launcher
    await waitForTestId(page, "home-chat-launcher");

    // Send a message to start the conversation
    await setChatInput(page, USER_MESSAGE);
    await page.getByTestId("submit-button").click();

    // Wait for navigation to the new conversation page
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    page.off("request", capturePayload);
    conversationId = getConversationIdFromURL(page);

    // ── Verify: the conversation launched from the active (ACP) profile ──
    // Conversations now launch from the active AgentProfile (#1571): the
    // payload carries `agent_profile_id` and omits `agent_settings` (the two
    // are mutually exclusive). Step 1 switched the active "default" profile to
    // ACP, so this launch uses it — the ACP reply token below confirms the ACP
    // agent actually ran.

    await test.step("verify conversation launched from the active agent profile", async () => {
      expect(
        capturedPayload,
        "POST /api/conversations payload was not captured",
      ).not.toBeNull();

      expect(
        capturedPayload!.agent_profile_id,
        `Expected agent_profile_id in payload, got: ${JSON.stringify(capturedPayload)}`,
      ).toBeTruthy();
      expect(
        capturedPayload!.agent_settings,
        "agent_settings must be omitted when launching from a profile",
      ).toBeUndefined();
    });

    // ── Verify: agent reply contains the ACP reply token ──

    await test.step("verify ACP agent reply appears in chat UI", async () => {
      try {
        await waitForNonUserMessageText(page, ACP_REPLY_TOKEN, 60_000);
      } catch (err) {
        // On failure, query the events API for diagnostic context
        let diag = "";
        try {
          const eventsResp = await request.get(
            `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId!)}/events/search`,
            {
              headers: { "X-Session-API-Key": SESSION_API_KEY },
              params: { limit: "50", sort_order: "TIMESTAMP_DESC" },
            },
          );
          if (eventsResp.ok()) {
            const body = (await eventsResp.json()) as { items?: unknown[] };
            const items = body.items ?? [];
            diag =
              `Events API returned ${items.length} events:\n` +
              items
                .map(
                  (e: any) =>
                    `  [${e.kind ?? "?"}] source=${e.source ?? "?"} ${JSON.stringify(
                      e.llm_message?.content ?? e.content ?? e.message ?? "",
                    ).slice(0, 120)}`,
                )
                .join("\n");
          } else {
            diag = `Events API returned ${eventsResp.status()}`;
          }
        } catch (diagErr) {
          diag = `Events API query failed: ${diagErr}`;
        }

        throw new Error(
          `ACP reply token "${ACP_REPLY_TOKEN}" not found in chat UI after 60s.\n` +
            `Conversation: ${conversationId}\n` +
            `ACP command: ${ACP_COMMAND_TEXT}\n` +
            `${diag}`,
          { cause: err },
        );
      }
    });

    // ── Verify: user message is visible in the chat UI ──

    await test.step("verify user message is visible in chat UI", async () => {
      await expect(
        page
          .locator('[data-testid="user-message"]')
          .filter({ hasText: USER_MESSAGE }),
      ).toBeVisible({ timeout: 5_000 });
    });

    // ── Verify: conversation appears in the sidebar ──

    await test.step("verify conversation appears in sidebar", async () => {
      const cardLinks = page.locator(
        `a[href*="/conversations/${conversationId}"]`,
      );
      await expect(cardLinks.first()).toBeVisible({ timeout: 10_000 });
    });

    // ── Verify: no error banners ──

    await test.step("verify no error banners", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });

  // ── Step 4: Resume the ACP conversation from the sidebar ────────────

  test("step 4: resume ACP conversation from sidebar after navigating away", async ({
    page,
  }) => {
    test.skip(!conversationId, "step 3 must complete first");

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the sidebar to load conversation cards
    const sidebarCards = page.locator('[data-testid="conversation-card"]');
    await expect(sidebarCards.first()).toBeVisible({ timeout: 15_000 });

    // Find the sidebar link to our conversation and click it
    const conversationLink = page.locator(
      `a[href*="/conversations/${conversationId}"]`,
    );
    await expect(conversationLink.first()).toBeVisible({ timeout: 10_000 });
    await conversationLink.first().click();

    // Wait for navigation back to the conversation page
    await waitForPath(page, /\/conversations\/.+/, 15_000);
    expect(page.url()).toContain(conversationId);

    // Verify the ACP agent's reply token is still visible after resume
    await test.step("verify ACP agent reply is still visible after resume", async () => {
      await waitForNonUserMessageText(page, ACP_REPLY_TOKEN, 15_000);
    });

    // Verify the user's original message is still visible
    await test.step("verify user message is still visible after resume", async () => {
      await expect(
        page
          .locator('[data-testid="user-message"]')
          .filter({ hasText: USER_MESSAGE }),
      ).toBeVisible({ timeout: 10_000 });
    });

    // Verify no error banners after resume
    await test.step("verify no error banners after resume", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
