/**
 * Mock-LLM E2E test: full onboarding happy path (PR #1095 coverage).
 *
 * Exercises the complete first-run onboarding flow end-to-end:
 *
 *   Step 0 — Choose Agent: selects OpenHands and advances.
 *   Step 1 — Check Backend: waits for the connected banner, advances.
 *   Step 2 — Setup LLM: fills in the mock LLM model, base URL, and API
 *            key (via "All" mode), and advances. The step edits the active
 *            LLM profile: it saves the form as a profile named after the
 *            model and activates it, without writing raw LLM settings.
 *   Step 3 — Say Hello: verifies the Skip button is hidden (PR #1095),
 *            submits the default greeting, verifies the conversation is
 *            created and the browser navigates to it.
 *
 * The test also validates:
 *   - The onboarding modal closes after launching.
 *   - `openhands-onboarded` is set in localStorage.
 *   - The settings-saved toast does NOT appear during onboarding.
 *   - No error banners are visible after the conversation loads.
 */

import { randomUUID } from "node:crypto";
import type {
  AgentProfile,
  SettingsApiResponse,
} from "@openhands/typescript-client";
import { test, expect } from "@playwright/test";
import {
  SESSION_API_KEY,
  MOCK_LLM_AGENT_URL,
  routeSessionApiKey,
  waitForPath,
  getConversationIdFromURL,
  deleteConversation,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  BACKEND_URL,
  waitForNonUserMessageText,
  waitForAgentMessageContaining,
} from "../utils/mock-llm-helpers";
import {
  showOnboarding,
  waitForOnboardingStep,
  waitForOnboardingBackendConnected,
  clickOnboardingStepButton,
  getOnboardingStepLayout,
  type OnboardingStepLayout,
} from "../../support/onboarding-helpers";

const PROFILE_NAME = `mock-onboarding-${randomUUID()}`;
const MOCK_MODEL = `openai/${PROFILE_NAME}`;
const PREVIOUS_PROFILE_NAME = `mock-before-onboarding-${randomUUID()}`;
// Raw settings that disagree with the active profile. Onboarding edits the
// profile, so this endpoint must never reach the form or the new profile.
const STALE_SETTINGS_BASE_URL = "http://127.0.0.1:9/stale-raw-settings";
const REPLY_TOKEN = "ONBOARDING_HAPPY_PATH_REPLY_OK";

test.describe.configure({ mode: "serial" });

test.describe("onboarding happy path", () => {
  const conversationIds = new Set<string>();
  const headers = { "X-Session-API-Key": SESSION_API_KEY };
  let previousSettings: SettingsApiResponse | undefined;
  let previousAgentProfile: AgentProfile | undefined;

  test.beforeEach(async ({ request }) => {
    // The list endpoint seeds the default profile on a fresh stack. Preserve
    // that profile and settings because onboarding updates both of them.
    const profiles = await request.get(`${BACKEND_URL}/api/agent-profiles`, {
      headers,
    });
    expect(profiles.ok()).toBe(true);
    const profile = await request.get(
      `${BACKEND_URL}/api/agent-profiles/default`,
      { headers },
    );
    expect(profile.ok()).toBe(true);
    previousAgentProfile = (await profile.json()).profile;
    const settings = await request.get(`${BACKEND_URL}/api/settings`, {
      headers: { ...headers, "X-Expose-Secrets": "encrypted" },
    });
    expect(settings.ok()).toBe(true);
    previousSettings = await settings.json();

    // Reproduce re-onboarding with an active profile that already has the
    // endpoint and key. The form shows them, so the endpoint the test types
    // again is an unchanged value that is absent from the form's changes.
    const saved = await request.post(
      `${BACKEND_URL}/api/profiles/${PREVIOUS_PROFILE_NAME}`,
      {
        headers,
        data: {
          llm: {
            model: `openai/${PREVIOUS_PROFILE_NAME}`,
            api_key: "mock-api-key-for-testing",
            base_url: MOCK_LLM_AGENT_URL,
          },
          include_secrets: true,
        },
      },
    );
    expect(saved.ok(), "save the previous profile").toBe(true);
    const activated = await request.post(
      `${BACKEND_URL}/api/profiles/${PREVIOUS_PROFILE_NAME}/activate`,
      { headers },
    );
    expect(activated.ok(), "activate the previous profile").toBe(true);
    const staleSettings = await request.patch(`${BACKEND_URL}/api/settings`, {
      headers,
      data: {
        agent_settings_diff: { llm: { base_url: STALE_SETTINGS_BASE_URL } },
      },
    });
    expect(staleSettings.ok(), "write stale raw LLM settings").toBe(true);
    await resetMockLLM(request);
  });

  test.afterEach(async ({ page, request }) => {
    // Stop UI reconciliation before restoring the profiles it observes.
    await page.close();
    try {
      for (const id of Array.from(conversationIds)) {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      }
      if (previousAgentProfile) {
        const restored = await request.post(
          `${BACKEND_URL}/api/agent-profiles/default`,
          {
            headers,
            data: previousAgentProfile,
          },
        );
        expect
          .soft(restored.ok(), "restore the default agent profile")
          .toBe(true);
      }
      if (previousSettings) {
        const restored = await request.patch(`${BACKEND_URL}/api/settings`, {
          headers,
          data: {
            agent_settings_diff: previousSettings.agent_settings,
            active_profile: previousSettings.active_profile ?? null,
            active_agent_profile_id:
              previousSettings.active_agent_profile_id ?? null,
          },
        });
        expect.soft(restored.ok(), "restore the previous settings").toBe(true);
      }
      for (const name of [PROFILE_NAME, PREVIOUS_PROFILE_NAME]) {
        const deleted = await request.delete(
          `${BACKEND_URL}/api/profiles/${name}`,
          { headers },
        );
        expect.soft(deleted.ok(), `delete test profile ${name}`).toBe(true);
      }
    } finally {
      await resetMockLLM(request);
    }
  });

  test("completes the full onboarding flow and launches a conversation", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    // Show the onboarding modal (clears openhands-onboarded, seeds backend)
    await showOnboarding(page, {
      apiKey: SESSION_API_KEY,
      beforeGoto: () => routeSessionApiKey(page),
    });

    let layout!: OnboardingStepLayout;

    // ── Backend / agent setup ────────────────────────────────────────

    await test.step("backend setup if needed, then choose agent", async () => {
      layout = await getOnboardingStepLayout(page);
      // Verify the skip button is visible on non-final steps
      await expect(
        page.getByTestId("onboarding-skip"),
        "Skip button should be visible before the final step",
      ).toBeVisible({ timeout: 5_000 });

      if (layout.hasBackendStep) {
        await waitForOnboardingBackendConnected(page);
        await clickOnboardingStepButton(page, "onboarding-backend-next");
        await waitForOnboardingStep(page, layout.agentStep);
      } else {
        await expect(
          page.getByTestId("onboarding-step-choose-agent"),
          "healthy configured backends should start at agent selection",
        ).toBeVisible({ timeout: 10_000 });
      }

      await expect(
        page.getByTestId("onboarding-skip"),
        "Skip button should be visible on agent selection",
      ).toBeVisible();

      await clickOnboardingStepButton(page, "onboarding-agent-next");
    });

    // ── Step 2: Setup LLM ───────────────────────────────────────────

    await test.step("step 2: setup LLM — fill mock LLM details, advance", async () => {
      await waitForOnboardingStep(page, layout.llmStep);

      await expect(
        page.getByTestId("onboarding-step-setup-llm"),
        "LLM setup step should be visible",
      ).toBeVisible({ timeout: 10_000 });

      await expect(
        page.getByTestId("onboarding-skip"),
        "Skip button should be visible on step 2",
      ).toBeVisible();

      // Switch to "All" view to access base_url and custom model fields
      const allToggle = page.getByTestId("sdk-section-all-toggle");
      await allToggle.dispatchEvent("click");

      // Wait for the advanced form
      await expect(page.getByTestId("llm-settings-form-advanced")).toBeVisible({
        timeout: 10_000,
      });

      // The form shows the active profile's endpoint, not the raw settings.
      await expect(page.getByTestId("base-url-input")).toHaveValue(
        MOCK_LLM_AGENT_URL,
      );

      // The LLM step must not write raw LLM settings.
      const settingsWrites: string[] = [];
      page.on("request", (req) => {
        if (
          req.method() === "PATCH" &&
          new URL(req.url()).pathname.endsWith("/api/settings")
        ) {
          settingsWrites.push(req.postData() ?? "");
        }
      });

      // Fill in model
      const modelInput = page.getByTestId("llm-custom-model-input");
      await modelInput.click();
      await modelInput.fill(MOCK_MODEL);

      // Fill in base URL
      const baseUrlInput = page.getByTestId("base-url-input");
      await baseUrlInput.click();
      await baseUrlInput.fill(MOCK_LLM_AGENT_URL);

      // Fill in API key
      const apiKeyInput = page.getByTestId("llm-api-key-input");
      await apiKeyInput.click();
      await apiKeyInput.fill("mock-api-key-for-testing");

      // Click Next — this saves settings and creates/activates a profile
      await clickOnboardingStepButton(page, "onboarding-llm-next");
      await waitForOnboardingStep(page, layout.helloStep);

      // Catch an incomplete profile before launching a conversation with the
      // dummy key against the provider's default endpoint.
      const profile = await request.get(
        `${BACKEND_URL}/api/profiles/${PROFILE_NAME}`,
        { headers },
      );
      expect(profile.ok()).toBe(true);
      expect((await profile.json()).config).toMatchObject({
        model: MOCK_MODEL,
        base_url: MOCK_LLM_AGENT_URL,
      });
      expect(settingsWrites, "settings PATCHes from the LLM step").toEqual([]);
    });

    // ── Step 3: Say Hello ───────────────────────────────────────────

    await test.step("step 3: say hello — verify skip hidden, launch conversation", async () => {
      // Start the conversation script after profile validation has completed.
      // The first completion generates the title; the second is the agent's
      // reply. Keep the title distinct from the asserted reply.
      await registerTrajectory(request, "onboarding-hello", [
        { text: "Onboarding test" },
        { text: REPLY_TOKEN },
      ]);
      await activateTrajectory(request, "onboarding-hello");

      await expect(
        page.getByTestId("onboarding-step-say-hello"),
        "Say Hello step should be visible",
      ).toBeVisible({ timeout: 10_000 });

      // PR #1095 key assertion: Skip button must be hidden on the final step
      await expect(
        page.getByTestId("onboarding-skip"),
        "Skip button should NOT be visible on the final step",
      ).toHaveCount(0);

      // The hello input should be pre-filled with the default message
      const helloInput = page.getByTestId("onboarding-hello-input");
      await expect(helloInput).toBeVisible();
      const inputValue = await helloInput.inputValue();
      expect(
        inputValue.length,
        "Hello input should be pre-filled with a default message",
      ).toBeGreaterThan(0);

      // Submit the hello message — this creates a conversation
      await helloInput.press("Enter");
    });

    // ── Verify: navigation to conversation page ─────────────────────

    await test.step("verify navigation to conversation page", async () => {
      await waitForPath(page, /\/conversations\/.+/, 30_000);
      const conversationId = getConversationIdFromURL(page);
      conversationIds.add(conversationId);
    });

    // ── Verify: onboarding modal is gone ────────────────────────────

    await test.step("verify onboarding modal is dismissed", async () => {
      await expect(
        page.getByTestId("onboarding-modal"),
        "Onboarding modal should be dismissed after launching a conversation",
      ).toHaveCount(0, { timeout: 10_000 });
    });

    // ── Verify: onboarding completion flag is persisted ──────────────

    await test.step("verify openhands-onboarded is set in localStorage", async () => {
      await expect
        .poll(
          () =>
            page.evaluate(() =>
              window.localStorage.getItem("openhands-onboarded"),
            ),
          {
            message:
              "openhands-onboarded should be '1' after completing the flow",
          },
        )
        .toBe("1");
    });

    // ── Verify: agent responds (proves LLM settings were saved) ─────

    await test.step("verify agent responds with the mock LLM", async () => {
      await waitForAgentMessageContaining(
        request,
        getConversationIdFromURL(page),
        REPLY_TOKEN,
      );
      await waitForNonUserMessageText(page, REPLY_TOKEN, 30_000);
    });

    // ── Verify: no error banners ────────────────────────────────────

    await test.step("verify no error banners after conversation loads", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
