import test, { expect } from "@playwright/test";
import {
  advanceOnboardingToLlmStep,
  ONBOARDING_BACKEND_STEP,
  routeOnboardingLlmCatalog,
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
  // should default users to the configured default provider and model.

  test("defaults the LLM setup step to OpenAI GPT-5.6 Sol", async ({
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
        // The local LLM step edits the active LLM profile, else the one the
        // `default` agent profile points at, and keeps that profile's model.
        // Hide both, which earlier specs may have set, so the step starts a
        // fresh profile with the default model, as on a first install.
        await page.route("**/api/profiles", async (route, req) => {
          if (req.method() !== "GET") {
            await route.fallback();
            return;
          }
          const response = await route.fetch();
          const body = await response.json();
          await route.fulfill({
            response,
            json: { ...body, active_profile: null },
          });
        });
        await page.route("**/api/agent-profiles", async (route, req) => {
          if (req.method() !== "GET") {
            await route.fallback();
            return;
          }
          const response = await route.fetch();
          const body = await response.json();
          await route.fulfill({
            response,
            json: {
              ...body,
              profiles: (body?.profiles ?? []).map(
                (profile: { name?: string }) =>
                  profile.name === "default"
                    ? { ...profile, llm_profile_ref: null }
                    : profile,
              ),
            },
          });
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
      "first-run onboarding should default to GPT-5.6 Sol",
    ).toHaveValue("gpt-5.6-sol", {
      timeout: 10_000,
    });
    await expect(
      page.getByTestId("openhands-account-help"),
      "OpenHands account helper should be hidden for OpenAI defaults",
    ).toBeHidden();
  });

  test("backend step radio buttons and disclosure controls meet WCAG 2.5.8 24px minimum target size", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.addInitScript(() => {
      window.localStorage.removeItem("openhands-onboarded");
      window.localStorage.removeItem("openhands-backends");
      window.localStorage.removeItem("openhands-active-backend");
      window.localStorage.setItem("analytics-consent", "false");
      window.localStorage.setItem("openhands-telemetry-consent", "denied");
      window.localStorage.setItem("openhands-telemetry-first-use", "true");
    });
    await routeSessionApiKey(page);
    await routeOnboardingLlmCatalog(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("onboarding-modal")).toBeVisible({
      timeout: 20_000,
    });
    await waitForOnboardingStep(page, ONBOARDING_BACKEND_STEP);

    for (const testId of [
      "onboarding-backend-kind-option-local",
      "onboarding-backend-kind-option-cloud",
    ]) {
      const control = page.getByTestId(testId);
      await expect(control).toBeVisible();
      const box = await control.evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { width: r.width, height: r.height };
      });
      expect(
        box.width,
        `${testId} width should be >= 24px`,
      ).toBeGreaterThanOrEqual(24);
      expect(
        box.height,
        `${testId} height should be >= 24px`,
      ).toBeGreaterThanOrEqual(24);
    }

    const advancedToggle = page.getByTestId(
      "onboarding-backend-advanced-toggle",
    );
    await expect(advancedToggle).toBeVisible();
    const advBox = await advancedToggle.evaluate((el) => {
      const r = el.getBoundingClientRect();
      return { width: r.width, height: r.height };
    });
    expect(
      advBox.width,
      "advanced toggle width should be >= 24px",
    ).toBeGreaterThanOrEqual(24);
    expect(
      advBox.height,
      "advanced toggle height should be >= 24px",
    ).toBeGreaterThanOrEqual(24);
  });
});
