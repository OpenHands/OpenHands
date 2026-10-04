/**
 * Mock-LLM E2E test: full UI-driven conversation with a scripted LLM backend.
 *
 * This test exercises the complete stack — from clicking around in the browser
 * through the real agent-server to a mock LLM server — without any real LLM
 * credentials. The mock LLM server uses openhands-sdk's TestLLM to return
 * scripted responses (tool calls and text).
 *
 * Flow:
 *   1. Navigate to Settings > LLM Profiles
 *   2. Create a new profile pointing at the mock LLM server
 *   3. Set the profile as active + verify settings API reflects the model
 *   4. Start a new conversation from the home page
 *   5. Send a user message and verify the agent responds correctly
 *   6. Verify via the events API that a terminal tool call was executed
 *   7. Verify the conversation appears in the sidebar
 *   8. Verify the user message is visible in chat
 *   9. Verify POST /api/conversations payload included worktree: true
 *  10. Resume the conversation from the sidebar after navigating away
 */

import { test, expect } from "@playwright/test";
import {
  BASH_TOKEN,
  REPLY_TOKEN,
  waitForAgentMessageContaining,
  MOCK_LLM_AGENT_URL,
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  waitForSuccessfulBashObservation,
  deleteConversation,
  resetMockLLM,
  setChatInput,
  ensureMockLLMAgentProfile,
} from "../utils/mock-llm-helpers";

const PROFILE_NAME = "mock-llm-e2e";
const MOCK_MODEL = "openai/mock-test-model";
const USER_MESSAGE = "Please run a quick terminal command and then reply.";

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM agent-server conversation", () => {
  const conversationIds = new Set<string>();
  /** Conversation ID from step 3, used by step 4 for resume verification. */
  let step3ConversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ page, request }) => {
    // Track any conversation we landed on
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    if (match?.[1]) conversationIds.add(decodeURIComponent(match[1]));

    // Clean up conversations — but skip step3ConversationId because step 4
    // needs it to verify conversation resume from the sidebar.
    for (const id of Array.from(conversationIds)) {
      if (id === step3ConversationId) continue;
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch {
        // best-effort cleanup
      }
    }
  });

  // Safety net: delete the shared step3 conversation after all tests complete.
  test.afterAll(async ({ request }) => {
    if (step3ConversationId) {
      try {
        await deleteConversation(request, step3ConversationId);
      } catch {
        // best-effort
      }
    }
  });

  // ── Step 1: Create LLM profile via the Settings UI ──────────────────

  test("step 1: create an LLM profile pointing at the mock LLM server", async ({
    page,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the profiles list to load
    await waitForTestId(page, "add-llm-profile");

    // Click "Add LLM Profile"
    await page.getByTestId("add-llm-profile").click();

    // Wait for the profile editor form to appear
    await waitForTestId(page, "profile-editor-title");

    // Fill in the profile name
    const nameInput = page.getByTestId("profile-name-input");
    await nameInput.click();
    await nameInput.fill(PROFILE_NAME);

    // Switch to "All" view to access base_url field
    await page.getByTestId("sdk-section-all-toggle").click();

    // Wait for the advanced form to render
    await waitForTestId(page, "llm-settings-form-advanced");

    // Fill in model name
    const modelInput = page.getByTestId("llm-custom-model-input");
    await modelInput.click();
    await modelInput.fill(MOCK_MODEL);

    // Fill in base URL pointing to our mock server.
    // Use MOCK_LLM_AGENT_URL — the URL the agent-server will use for
    // inference calls. In Docker this may differ from the host-local URL.
    const baseUrlInput = page.getByTestId("base-url-input");
    await baseUrlInput.click();
    await baseUrlInput.fill(MOCK_LLM_AGENT_URL);

    // Fill in a fake API key (mock server doesn't validate it)
    const apiKeyInput = page.getByTestId("llm-api-key-input");
    await apiKeyInput.click();
    await apiKeyInput.fill("mock-api-key-for-testing");

    // Save the profile
    await page.getByTestId("save-profile-btn").click();

    // Wait to return to the profiles list
    await waitForTestId(page, "add-llm-profile");

    // Verify the profile appears in the list
    const profileRows = page.getByTestId("profile-row");
    const profileTexts = await profileRows.allTextContents();
    const hasProfile = profileTexts.some((text) =>
      text.includes(PROFILE_NAME),
    );
    expect(hasProfile, `Profile "${PROFILE_NAME}" should appear in the list`).toBe(true);
  });

  // ── Step 2: Set the profile as active ───────────────────────────────

  test("step 2: activate the mock-llm profile and verify settings API", async ({
    page,
    request,
  }) => {
    await routeSessionApiKey(page);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "add-llm-profile");

    // Find the profile row containing our profile name
    const profileRows = page.getByTestId("profile-row");
    const rowCount = await profileRows.count();
    let targetRow: ReturnType<typeof profileRows.nth> | null = null;

    for (let i = 0; i < rowCount; i++) {
      const row = profileRows.nth(i);
      const text = await row.textContent();
      if (text?.includes(PROFILE_NAME)) {
        targetRow = row;
        break;
      }
    }
    expect(targetRow, `Could not find profile row for "${PROFILE_NAME}"`).not.toBeNull();

    // Open the actions menu for this profile
    await targetRow!.getByTestId("profile-menu-trigger").click();
    await waitForTestId(page, "profile-actions-menu");

    // Click "Set as active" — with client-side reconciliation
    // (useEnsureActiveProfile) a freshly-created keyed profile may already be
    // auto-activated, which disables this item. Only click when it isn't active
    // yet; either way the badge poll below verifies the end state.
    const setActive = page.getByTestId("profile-set-active");
    if (await setActive.isEnabled()) {
      await setActive.click();
    } else {
      await page.keyboard.press("Escape");
    }

    // Verify the "Active" badge appears on our profile.
    // Poll with reload instead of a fixed timeout — the mutation may take
    // more than 1s to persist on a loaded CI runner. The poll also
    // RE-ATTEMPTS the activation when the badge is still missing: the
    // "Set as active" PATCH above can be lost when it races the
    // useEnsureActiveProfile reconciliation write, and a read-only poll
    // would then time out with no profile active at all.
    await expect
      .poll(
        async () => {
          await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
          await waitForTestId(page, "add-llm-profile");
          const rows = page.getByTestId("profile-row");
          const count = await rows.count();
          for (let i = 0; i < count; i++) {
            const row = rows.nth(i);
            const text = await row.textContent();
            if (text?.includes(PROFILE_NAME)) {
              if (
                (await row.getByTestId("profile-active-badge").count()) > 0
              ) {
                return true;
              }
              // Badge absent — re-attempt activation before the next poll.
              await row.getByTestId("profile-menu-trigger").click();
              await waitForTestId(page, "profile-actions-menu");
              const retrySetActive = page.getByTestId("profile-set-active");
              if (await retrySetActive.isEnabled()) {
                await retrySetActive.click();
              } else {
                await page.keyboard.press("Escape");
              }
              return false;
            }
          }
          return false;
        },
        {
          message: `Profile "${PROFILE_NAME}" should have an "Active" badge`,
          timeout: 25_000,
          intervals: [1_000, 2_000, 3_000],
        },
      )
      .toBe(true);

    // Verify the settings API now reflects the activated profile's LLM config
    await test.step("verify settings API reflects the active profile's model", async () => {
      const settingsResp = await request.get(`${BACKEND_URL}/api/settings`, {
        headers: {
          "X-Session-API-Key": SESSION_API_KEY,
          "X-Expose-Secrets": "encrypted",
        },
      });
      expect(settingsResp.ok(), `GET /api/settings returned ${settingsResp.status()}`).toBe(true);
      const settings = await settingsResp.json();
      const llmModel = settings?.agent_settings?.llm?.model;
      expect(
        llmModel,
        `Expected settings llm.model="${MOCK_MODEL}" but got "${llmModel}"`,
      ).toBe(MOCK_MODEL);

      const llmBaseUrl = settings?.agent_settings?.llm?.base_url;
      expect(
        llmBaseUrl,
        `Expected settings llm.base_url="${MOCK_LLM_AGENT_URL}" but got "${llmBaseUrl}"`,
      ).toBe(MOCK_LLM_AGENT_URL);
    });

    // Point the active agent profile at this LLM profile so the home composer
    // is unblocked in step 3. Conversations launch from the active AGENT
    // profile (#1571), whose `llm_profile_ref` — not the active LLM profile —
    // gates the composer; onboarding does this for real users, but this spec
    // seeds `openhands-onboarded` and configures the LLM directly.
    await ensureMockLLMAgentProfile(request, PROFILE_NAME);
  });

  // ── Step 3: Start a conversation and verify the mock agent responds ─

  test("step 3: run a conversation with the mock LLM", async ({
    page,
    request,
  }) => {
    // Reset the mock LLM to its default trajectory. Other test suites
    // (e.g. automation tests) may have activated a different trajectory
    // that wasn't fully consumed due to earlier failures.
    await resetMockLLM(request);

    // Steps 1+2 already configured and verified the profile via the UI
    // (including the "Active" badge check). No API pre-check needed.

    // Passively observe POST /api/conversations to capture the request body.
    // Using page.on('request') instead of page.route() avoids conflicts with
    // the routeSessionApiKey interceptor (Playwright routes are LIFO and only
    // one handler can call continue/fulfill per request).
    let capturedConversationPayload: Record<string, unknown> | null = null;
    const captureConversationPayload = (req: import("@playwright/test").Request) => {
      if (
        req.method() === "POST" &&
        new URL(req.url()).pathname === "/api/conversations"
      ) {
        try {
          capturedConversationPayload = req.postDataJSON();
        } catch {
          // non-JSON body — leave null
        }
      }
    };
    page.on("request", captureConversationPayload);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the home page chat launcher to appear
    await waitForTestId(page, "home-chat-launcher");

    // Type a message into the home-page chat input, then submit.
    // This creates the conversation AND sends the first user message in
    // one step — matching how real users interact with the home page.
    // IMPORTANT: Do NOT include BASH_TOKEN or REPLY_TOKEN in the user
    // message. The mock LLM ignores the prompt entirely (TestLLM pops
    // scripted responses from a deque), so the prompt text is irrelevant.
    // Keeping the tokens out of the user bubble lets us assert they appear
    // *only* in agent output.

    await setChatInput(page, USER_MESSAGE);

    // Click the submit button — this triggers conversation creation
    await page.getByTestId("submit-button").click();

    // Wait for navigation to the new conversation page
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    page.off("request", captureConversationPayload);
    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);
    step3ConversationId = conversationId;

    // ── Verify: POST /api/conversations payload contained worktree: true ──

    await test.step("verify worktree:true in conversation creation payload", async () => {
      expect(
        capturedConversationPayload,
        "POST /api/conversations payload was not captured — " +
        "the page.on('request') listener may have missed the request",
      ).not.toBeNull();
      expect(
        capturedConversationPayload?.worktree,
        `Expected worktree=true in payload, got: ${JSON.stringify(capturedConversationPayload?.worktree)}`,
      ).toBe(true);
    });

    // ── Verify: bash tool was executed (via bash events API) ──

    await test.step("verify bash tool execution via bash events API", async () => {
      await waitForSuccessfulBashObservation(request, conversationId);
    });

    // ── Verify: agent reply contains REPLY_TOKEN (via conversation events API) ──

    await test.step("verify agent reply via conversation events API", async () => {
      await waitForAgentMessageContaining(
        request, conversationId, REPLY_TOKEN, 30_000,
      );
    });

    // ── Verify: the agent's final reply token appears in the chat UI ──

    await test.step("verify agent reply token appears in chat UI", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 30_000);
    });

    // ── Verify: user message is visible in the chat UI ──

    await test.step("verify user message is visible in chat UI", async () => {
      const userMessages = page.locator('[data-testid="user-message"]');
      await expect(userMessages.first()).toBeVisible({ timeout: 5_000 });
      const allUserText = await userMessages.allTextContents();
      const hasUserMessage = allUserText.some((text) =>
        text.includes(USER_MESSAGE),
      );
      expect(
        hasUserMessage,
        `User message "${USER_MESSAGE}" should be visible in a user-message element. ` +
        `Found: ${allUserText.map((t) => t.slice(0, 80)).join(" | ")}`,
      ).toBe(true);
    });

    // ── Verify: conversation appears in the sidebar ──

    await test.step("verify conversation appears in sidebar", async () => {
      // The sidebar renders conversation cards with data-testid="conversation-card".
      // After creating a conversation, at least one card should be visible, and
      // clicking it should link to our conversation's URL.
      const sidebarCards = page.locator('[data-testid="conversation-card"]');
      await expect(sidebarCards.first()).toBeVisible({ timeout: 10_000 });

      // Verify at least one sidebar card links to our conversation
      const cardLinks = page.locator(
        `a[href*="/conversations/${conversationId}"]`,
      );
      await expect(cardLinks.first()).toBeVisible({ timeout: 5_000 });
    });

    // ── Verify: no error banners are visible ──

    await test.step("verify no error banners", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      // No .catch() — if the banner IS visible, this step must fail the test.
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });

  });

  // ── Step 4: Resume the conversation from the sidebar ────────────────

  test("step 4: resume conversation from sidebar after navigating away", async ({
    page,
  }) => {
    // This step depends on the conversation created in step 3.
    // If step 3 failed, skip this test instead of failing with a confusing error.
    test.skip(!step3ConversationId, "step 3 must complete first");

    await routeSessionApiKey(page);

    // Navigate away to the home page
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the sidebar to load conversation cards
    const sidebarCards = page.locator('[data-testid="conversation-card"]');
    await expect(sidebarCards.first()).toBeVisible({ timeout: 15_000 });

    // Find the sidebar link to our conversation and click it
    const conversationLink = page.locator(
      `a[href*="/conversations/${step3ConversationId}"]`,
    );
    await expect(conversationLink.first()).toBeVisible({ timeout: 10_000 });
    await conversationLink.first().click();

    // Wait for navigation back to the conversation page
    await waitForPath(page, /\/conversations\/.+/, 15_000);
    expect(page.url()).toContain(step3ConversationId);

    // Verify the agent's reply token is still visible after resume
    await test.step("verify agent reply is still visible after resume", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 15_000);
    });

    // Verify the user's original message is still visible
    await test.step("verify user message is still visible after resume", async () => {
      await expect(
        page.locator('[data-testid="user-message"]').filter({ hasText: USER_MESSAGE }),
      ).toBeVisible({ timeout: 10_000 });
    });

    // Verify no error banners after resume
    await test.step("verify no error banners after resume", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
