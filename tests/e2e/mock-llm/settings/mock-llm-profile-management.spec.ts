/**
 * Mock-LLM E2E tests: LLM profile management regressions.
 *
 * Covers two scenarios that previously had no end-to-end guard:
 *
 *   1. Active profile deletion + reconciliation:
 *      The active LLM profile IS deletable (the PR #1127 disable-guard was
 *      removed). Deleting it must not strand the app: useEnsureActiveProfile
 *      promotes a remaining profile to active in local mode, so a usable LLM
 *      is always selected. This test verifies delete is enabled and that the
 *      remaining profile becomes active.
 *
 *   2. Same-model profile identity (PR #1123):
 *      When two profiles share the same underlying model, the chat
 *      header must display the correct profile name — the one the
 *      user selected, not the first alphabetical match. The fix
 *      stamps the active profile name on client-side conversation
 *      metadata at creation and on per-conversation switches.
 *
 *   3. OpenHands provider hidden base_url preservation:
 *      A base_url typed in Advanced view is still real profile data after
 *      switching to Basic. Re-saving from Basic without changing the model
 *      must preserve that hidden value.
 */

import { test, expect } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  deleteConversation,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  setChatInput,
  waitForPath,
  createProfileViaUI,
  deleteProfileIfExists,
  activateProfileViaUI,
  ensureMockLLMAgentProfile,
} from "../utils/mock-llm-helpers";

const MOCK_MODEL = "openai/mock-test-model";

/**
 * Read-only helper: fetch a profile's persisted config via the API.
 * Used to verify that UI-driven saves persisted the expected values.
 */
async function getProfileConfig(
  request: import("@playwright/test").APIRequestContext,
  name: string,
): Promise<Record<string, unknown>> {
  const resp = await request.get(
    `${BACKEND_URL}/api/profiles/${encodeURIComponent(name)}`,
    {
      headers: {
        "X-Session-API-Key": SESSION_API_KEY,
        "X-Expose-Secrets": "encrypted",
      },
    },
  );
  expect(resp.ok(), `GET /api/profiles/${name}: ${resp.status()}`).toBe(true);
  const data = await resp.json();
  return (data.config ?? {}) as Record<string, unknown>;
}

test.describe.configure({ mode: "serial" });

// ═══════════════════════════════════════════════════════════════════════
// Test 1 — Active profile deletion guard (PR #1127)
// ═══════════════════════════════════════════════════════════════════════

test.describe("active profile deletion + reconciliation", () => {
  const ACTIVE_PROFILE = "deletion-guard-active";
  const INACTIVE_PROFILE = "deletion-guard-inactive";

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterAll(async ({ browser }) => {
    // Best-effort cleanup via UI
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await routeSessionApiKey(page);
      await ensureMockLLMAgentProfile(page.request);
      await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
      await dismissAnalyticsModal(page);
      await waitForTestId(page, "add-llm-profile");
      await deleteProfileIfExists(page, ACTIVE_PROFILE);
      await deleteProfileIfExists(page, INACTIVE_PROFILE);
    } catch {
      // best-effort
    } finally {
      await page.close();
    }
  });

  test("active profile is deletable and reconciliation activates another profile", async ({
    page,
  }) => {
    // ── Setup: create two profiles via the UI, activate one ──
    await routeSessionApiKey(page);
    await ensureMockLLMAgentProfile(page.request);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "add-llm-profile");

    // Clean up any leftover profiles from prior runs
    await deleteProfileIfExists(page, ACTIVE_PROFILE);
    await deleteProfileIfExists(page, INACTIVE_PROFILE);

    // Create both profiles through the Settings UI
    await createProfileViaUI(page, {
      profileName: ACTIVE_PROFILE,
      model: MOCK_MODEL,
    });
    await createProfileViaUI(page, {
      profileName: INACTIVE_PROFILE,
      model: MOCK_MODEL,
    });

    // Activate the first profile through the UI
    await activateProfileViaUI(page, ACTIVE_PROFILE);

    const rowFor = async (name: string) => {
      const rows = page.getByTestId("profile-row");
      const count = await rows.count();
      for (let i = 0; i < count; i++) {
        const row = rows.nth(i);
        if ((await row.textContent())?.includes(name)) return row;
      }
      return null;
    };

    // ── Delete is now enabled on the active profile (the #1127 guard was
    //    removed; useEnsureActiveProfile keeps a profile active instead) ──
    await test.step("active profile: delete button is enabled", async () => {
      const activeRow = await rowFor(ACTIVE_PROFILE);
      expect(
        activeRow,
        `Could not find profile row for "${ACTIVE_PROFILE}"`,
      ).not.toBeNull();

      await activeRow!.getByTestId("profile-menu-trigger").click();
      await waitForTestId(page, "profile-actions-menu");

      const deleteButton = page.getByTestId("profile-delete");
      await expect(
        deleteButton,
        "Delete button should be present in the menu",
      ).toBeVisible();
      await expect(
        deleteButton,
        "Delete should be enabled for the active profile",
      ).toBeEnabled();

      // Edit and Set-as-active should still be present
      await expect(page.getByTestId("profile-edit")).toBeVisible();
      await expect(page.getByTestId("profile-set-active")).toBeVisible();

      await page.keyboard.press("Escape");
    });

    // ── Delete is enabled on an inactive profile too ──
    await test.step("inactive profile: delete button is enabled", async () => {
      const inactiveRow = await rowFor(INACTIVE_PROFILE);
      expect(
        inactiveRow,
        `Could not find profile row for "${INACTIVE_PROFILE}"`,
      ).not.toBeNull();

      await inactiveRow!.getByTestId("profile-menu-trigger").click();
      await waitForTestId(page, "profile-actions-menu");
      await expect(page.getByTestId("profile-delete")).toBeEnabled();
      await page.keyboard.press("Escape");
    });

    // ── Deleting the active profile reconciles to the remaining one ──
    await test.step("deleting the active profile activates the remaining profile", async () => {
      const activeRow = await rowFor(ACTIVE_PROFILE);
      expect(activeRow).not.toBeNull();

      await activeRow!.getByTestId("profile-menu-trigger").click();
      await waitForTestId(page, "profile-actions-menu");
      await page.getByTestId("profile-delete").click();

      // Confirm in the delete modal.
      await page.getByTestId("delete-profile-confirm").click();

      // useEnsureActiveProfile re-activates the only remaining profile. Poll
      // with reload — the delete + activate mutations may take a moment on CI.
      await expect
        .poll(
          async () => {
            await page.goto("/settings/llm", {
              waitUntil: "domcontentloaded",
            });
            await waitForTestId(page, "add-llm-profile");
            const remaining = await rowFor(INACTIVE_PROFILE);
            if (!remaining) return false;
            // The deleted profile must be gone, and reconciliation must keep
            // *some* profile active (the "always have an active profile"
            // guarantee). We don't assert it's INACTIVE_PROFILE specifically —
            // other profiles may linger on the shared agent-server and
            // useEnsureActiveProfile activates the first keyed one.
            const goneRow = await rowFor(ACTIVE_PROFILE);
            const activeBadges = await page
              .getByTestId("profile-active-badge")
              .count();
            return goneRow === null && activeBadges > 0;
          },
          {
            message: `"${INACTIVE_PROFILE}" should become active after deleting "${ACTIVE_PROFILE}"`,
            timeout: 15_000,
            intervals: [1_000, 2_000, 3_000],
          },
        )
        .toBe(true);
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Test 2 — Same-model profile identity (PR #1123)
// ═══════════════════════════════════════════════════════════════════════

test.describe("same-model profile identity", () => {
  // Two profiles with DIFFERENT names but the SAME underlying model.
  // Alphabetically, PROFILE_ALPHA < PROFILE_BETA — before the fix,
  // the UI would always show PROFILE_ALPHA regardless of which was active.
  const PROFILE_ALPHA = "aaa-profile-alpha";
  const PROFILE_BETA = "zzz-profile-beta";
  const SHARED_MODEL = "openai/mock-test-model";
  const REPLY_TOKEN = "PROFILE_IDENTITY_REPLY_OK";

  const conversationIds = new Set<string>();

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch {
        // best-effort
      }
    }
  });

  test.afterAll(async ({ request, browser }) => {
    // Best-effort cleanup via UI
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await routeSessionApiKey(page);
      await ensureMockLLMAgentProfile(page.request);
      await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
      await dismissAnalyticsModal(page);
      await waitForTestId(page, "add-llm-profile");
      await deleteProfileIfExists(page, PROFILE_ALPHA);
      await deleteProfileIfExists(page, PROFILE_BETA);
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

  test("chat header shows the correct profile when two profiles share the same model", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    // ── Setup: create both profiles with the same model via the UI,
    //    then activate BETA through the profile menu ──
    await routeSessionApiKey(page);
    await ensureMockLLMAgentProfile(page.request);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "add-llm-profile");

    await deleteProfileIfExists(page, PROFILE_ALPHA);
    await deleteProfileIfExists(page, PROFILE_BETA);

    await createProfileViaUI(page, {
      profileName: PROFILE_ALPHA,
      model: SHARED_MODEL,
    });
    await createProfileViaUI(page, {
      profileName: PROFILE_BETA,
      model: SHARED_MODEL,
    });
    await activateProfileViaUI(page, PROFILE_BETA);

    // Register a trajectory for the conversation.
    // Turn 0 is padding: the agent-server makes an internal LLM call
    // (condenser/skill-analysis) before the agent's main loop starts.
    await registerTrajectory(request, "profile-identity", [
      { text: "" }, // padding for internal condenser call
      { text: REPLY_TOKEN },
    ]);
    await activateTrajectory(request, "profile-identity");

    // ── Start a conversation ──
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    await setChatInput(page, "Test profile identity.");
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    // Wait for the agent to reply so the conversation is fully established
    await waitForNonUserMessageText(page, REPLY_TOKEN, 30_000);

    // ── Verify: profile switcher shows BETA, not ALPHA ──
    await test.step("profile switcher shows the correct profile name", async () => {
      // In a local OpenHands conversation the chat input renders the
      // in-conversation LLM-profile picker, whose label is the active
      // profile name (replaces the former switch-profile-button).
      const profilePicker = page.getByTestId("chat-input-llm-profile");
      await expect(profilePicker).toBeVisible({ timeout: 10_000 });
      // The picker's visible text should contain PROFILE_BETA.
      // Before the fix (PR #1123), it would show PROFILE_ALPHA because
      // profiles were matched by model name and .find() returned the
      // first alphabetical match.
      await expect(profilePicker).toContainText(PROFILE_BETA, {
        timeout: 10_000,
      });
    });

    // ── Verify: profile identity survives a page reload ──
    await test.step("profile identity persists after page reload", async () => {
      await page.reload({ waitUntil: "domcontentloaded" });

      // Re-wait for the conversation to load
      await waitForNonUserMessageText(page, REPLY_TOKEN, 30_000);

      const profilePicker = page.getByTestId("chat-input-llm-profile");
      await expect(profilePicker).toBeVisible({ timeout: 10_000 });
      await expect(
        profilePicker,
        "Profile identity should persist across page reloads",
      ).toContainText(PROFILE_BETA, { timeout: 10_000 });
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Test 3 — OpenHands provider hidden base_url preservation
// ═══════════════════════════════════════════════════════════════════════

test.describe("OpenHands provider hidden base_url preservation", () => {
  // The public OpenHands provider model remains the profile identity, but a
  // custom Advanced base_url is still user data. A same-model Basic save must
  // not wipe it just because the field is invisible in that view.
  const OPENHANDS_PROFILE = "openhands-basic-save-test";
  const OPENHANDS_MODEL = "openhands/claude-opus-4-5-20251101";
  const LEGACY_TRANSPORT_MODEL = "litellm_proxy/claude-opus-4-5-20251101";
  const EXPECTED_OPENHANDS_MODELS = [OPENHANDS_MODEL, LEGACY_TRANSPORT_MODEL];
  const CUSTOM_BASE_URL = "https://custom-openhands-proxy.example/v1";

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterAll(async ({ browser }) => {
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await routeSessionApiKey(page);
      await ensureMockLLMAgentProfile(page.request);
      await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
      await dismissAnalyticsModal(page);
      await waitForTestId(page, "add-llm-profile");
      await deleteProfileIfExists(page, OPENHANDS_PROFILE);
    } catch {
      // best-effort
    } finally {
      await page.close();
    }
  });

  test("re-saving an OpenHands profile from Basic view preserves hidden base_url", async ({
    page,
    request,
  }) => {
    // ── Setup: create a public OpenHands profile with a custom base_url through
    // Advanced view. The value becomes hidden after switching to Basic, but it
    // is still part of the profile unless the model changes. ──
    await routeSessionApiKey(page);
    await ensureMockLLMAgentProfile(page.request);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "add-llm-profile");

    await deleteProfileIfExists(page, OPENHANDS_PROFILE);
    await createProfileViaUI(page, {
      profileName: OPENHANDS_PROFILE,
      model: OPENHANDS_MODEL,
      baseUrl: CUSTOM_BASE_URL,
    });

    await test.step("open profile in edit mode", async () => {
      const profileRows = page.getByTestId("profile-row");
      const rowCount = await profileRows.count();
      let targetRow: ReturnType<typeof profileRows.nth> | null = null;

      for (let i = 0; i < rowCount; i++) {
        const row = profileRows.nth(i);
        const text = await row.textContent();
        if (text?.includes(OPENHANDS_PROFILE)) {
          targetRow = row;
          break;
        }
      }
      expect(
        targetRow,
        `Could not find profile row for "${OPENHANDS_PROFILE}"`,
      ).not.toBeNull();

      await targetRow!.getByTestId("profile-menu-trigger").click();
      await waitForTestId(page, "profile-actions-menu");
      await page.getByTestId("profile-edit").click();

      await expect(page.getByTestId("profile-name-input")).toHaveValue(
        OPENHANDS_PROFILE,
        { timeout: 10_000 },
      );
    });

    await test.step("switch to Basic view and save without changing model", async () => {
      // Click the Basic toggle explicitly so the save path exercises the view
      // that hides base_url without changing the selected model.
      const basicToggle = page.getByTestId("sdk-section-basic-toggle");
      if (await basicToggle.isVisible().catch(() => false)) {
        await basicToggle.click();
      }

      const saveButton = page.getByTestId("save-profile-btn");
      await expect(saveButton).toBeEnabled({ timeout: 10_000 });
      await saveButton.click();

      await waitForTestId(page, "add-llm-profile");
    });

    // ── Verify: Basic-tab save preserved the hidden custom base_url ──
    await test.step("verify base_url is preserved after save", async () => {
      const config = await getProfileConfig(request, OPENHANDS_PROFILE);
      expect(
        config.base_url,
        "Basic-tab re-save without a model change must not clear hidden base_url",
      ).toBe(CUSTOM_BASE_URL);
      expect(EXPECTED_OPENHANDS_MODELS).toContain(config.model);
    });

    await test.step("profile survives page reload", async () => {
      await page.reload({ waitUntil: "domcontentloaded" });
      await waitForTestId(page, "add-llm-profile");

      // Re-read via API to confirm persistence is durable
      const config = await getProfileConfig(request, OPENHANDS_PROFILE);
      expect(config.base_url).toBe(CUSTOM_BASE_URL);
      expect(EXPECTED_OPENHANDS_MODELS).toContain(config.model);
    });
  });
});
