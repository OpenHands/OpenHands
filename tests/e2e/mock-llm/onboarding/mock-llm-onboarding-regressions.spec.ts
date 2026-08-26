import test, { expect } from "@playwright/test";
import {
  advanceOnboardingToLlmStep,
  ONBOARDING_BACKEND_STEP,
  showOnboarding,
  waitForOnboardingStep,
} from "../../support/onboarding-helpers";
import { routeSessionApiKey, SESSION_API_KEY } from "../utils/mock-llm-helpers";

test.describe.configure({ mode: "serial" });

test.describe("onboarding recent regressions", () => {
  // Regression coverage for #1085 / PR #1100: errant outside
  // interactions must not permanently mark onboarding complete.

  test("keeps the modal open on backdrop click and Escape", async ({
    page,
  }) => {
    await showOnboarding(page, {
      apiKey: SESSION_API_KEY,
      beforeGoto: () => routeSessionApiKey(page),
    });

    // Exercise the original first-load path before any onboarding step
    // interaction.
    await page.mouse.click(8, 8);
    await page.keyboard.press("Escape");

    await expect(
      page.getByTestId("onboarding-modal"),
      "onboarding modal should ignore backdrop clicks and Escape",
    ).toBeVisible();
    await waitForOnboardingStep(page, ONBOARDING_BACKEND_STEP);
    await expect
      .poll(
        () =>
          page.evaluate(() =>
            window.localStorage.getItem("openhands-onboarded"),
          ),
        {
          message:
            "onboarding should not be marked complete by outside interactions",
        },
      )
      .toBeNull();

    await page.getByTestId("onboarding-skip").click();
    await expect(
      page.getByTestId("onboarding-modal"),
      "skip should close the onboarding modal",
    ).toHaveCount(0);
    await expect
      .poll(
        () =>
          page.evaluate(() =>
            window.localStorage.getItem("openhands-onboarded"),
          ),
        { message: "skip should persist onboarding completion" },
      )
      .toBe("1");
  });

  // Regression coverage for #1077 / PR #1089: first-run LLM setup
  // should not default users to the OpenHands provider.

  test("defaults the LLM setup step to OpenAI GPT-5.5", async ({
    page,
  }) => {
    await showOnboarding(page, {
      apiKey: SESSION_API_KEY,
      beforeGoto: async () => {
        await routeSessionApiKey(page);
        // Intercept GET /api/settings so the LLM form sees a clean
        // base_url, forcing basic view mode regardless of what earlier
        // specs configured. Registered AFTER routeSessionApiKey so
        // Playwright's LIFO matching picks this up first for settings.
        await page.route("**/api/settings", async (route, req) => {
          if (req.method() !== "GET") {
            await route.fallback();
            return;
          }
          const response = await route.fetch();
          const body = await response.json();
          if (body?.agent_settings?.llm) {
            body.agent_settings.llm.base_url = null;
          }
          await route.fulfill({ response, json: body });
        });
      },
    });
    await advanceOnboardingToLlmStep(page);

    const providerInput = page.locator('input[name="llm-provider-input"]');
    const modelInput = page.locator('input[name="llm-model-input"]');

    await expect(
      providerInput,
      "first-run onboarding should default to the OpenAI provider",
    ).toHaveValue("OpenAI", { timeout: 10_000 });
    // The model input displays the model ID without the provider prefix.
    await expect(
      modelInput,
      "first-run onboarding should default to GPT-5.5",
    ).toHaveValue("gpt-5.5", {
      timeout: 10_000,
    });
    await expect(
      page.getByTestId("openhands-account-help"),
      "OpenHands account helper should stay hidden for OpenAI defaults",
    ).toHaveCount(0);
  });
});
