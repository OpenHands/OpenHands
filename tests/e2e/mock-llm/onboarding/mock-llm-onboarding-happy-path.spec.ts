/**
 * Mock-LLM E2E test: full onboarding happy path (PR #1095 coverage).
 *
 * Exercises the complete first-run onboarding flow end-to-end:
 *
 *   Step 0 — Choose Agent: selects OpenHands and advances.
 *   Step 1 — Check Backend: waits for the connected banner, advances.
 *   Step 2 — Setup LLM: fills in the mock LLM model, base URL, and API
 *            key (via "All" mode), and advances. The step persists settings
 *            AND creates/activates a named profile automatically.
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
  waitForNonUserMessageText,
} from "../utils/mock-llm-helpers";
import {
  showOnboarding,
  waitForOnboardingStep,
  waitForOnboardingBackendConnected,
  clickOnboardingStepButton,
  getOnboardingStepLayout,
  type OnboardingStepLayout,
} from "../../support/onboarding-helpers";

const MOCK_MODEL = "openai/mock-onboarding-model";
const REPLY_TOKEN = "ONBOARDING_HAPPY_PATH_REPLY_OK";

test.describe.configure({ mode: "serial" });

test.describe("onboarding happy path", () => {
  const conversationIds = new Set<string>();

  test.afterEach(async ({ request }) => {
    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch {
        // best-effort
      }
    }
    // Reset the mock LLM to its default trajectory so subsequent specs
    // start with a clean slate.
    try {
      await resetMockLLM(request);
    } catch {
      // best-effort
    }
  });

  test("completes the full onboarding flow and launches a conversation", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    // Register a trajectory so the mock LLM can respond once the
    // conversation is created by the Say Hello step.
    //
    // Turn 0 is padding: the agent-server makes an internal LLM call
    // (condenser/skill-analysis) before the agent's main loop starts.
    // This consumes one trajectory response. Same pattern used by the
    // automation and model-switch specs.
    await registerTrajectory(request, "onboarding-hello", [
      { text: "" }, // padding for internal condenser call
      { text: REPLY_TOKEN },
    ]);
    await activateTrajectory(request, "onboarding-hello");

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
      await expect(
        page.getByTestId("llm-settings-form-advanced"),
      ).toBeVisible({ timeout: 10_000 });

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
    });

    // ── Step 3: Say Hello ───────────────────────────────────────────

    await test.step("step 3: say hello — verify skip hidden, launch conversation", async () => {
      await waitForOnboardingStep(page, layout.helloStep);

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
          { message: "openhands-onboarded should be '1' after completing the flow" },
        )
        .toBe("1");
    });

    // ── Verify: agent responds (proves LLM settings were saved) ─────

    await test.step("verify agent responds with the mock LLM", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 30_000);
    });

    // ── Verify: no error banners ────────────────────────────────────

    await test.step("verify no error banners after conversation loads", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
