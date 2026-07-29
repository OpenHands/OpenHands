import { expect, type Page } from "@playwright/test";

export const ONBOARDING_BACKEND_STEP = 0;
export const ONBOARDING_AGENT_STEP = 1;
export const ONBOARDING_LLM_STEP = 2;
export const ONBOARDING_HELLO_STEP = 3;

export type OnboardingStepLayout = {
  hasBackendStep: boolean;
  agentStep: number;
  llmStep: number;
  helloStep: number;
};

export async function routeOnboardingLlmCatalog(page: Page) {
  await page.route("**/api/llm/models/verified", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: {
          anthropic: ["claude-opus-4-8"],
          openai: ["gpt-5.5"],
          openhands: ["claude-opus-4-5-20251101"],
        },
      }),
    });
  });

  await page.route("**/api/llm/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          "anthropic/claude-opus-4-8",
          "openai/gpt-5.5",
          "openhands/claude-opus-4-5-20251101",
        ],
      }),
    });
  });

  await page.route("**/api/llm/providers", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        providers: ["anthropic", "openai", "openhands"],
      }),
    });
  });
}

type ShowOnboardingOptions = {
  apiKey: string;
  beforeGoto?: () => Promise<void>;
};

export async function showOnboarding(
  page: Page,
  { apiKey, beforeGoto }: ShowOnboardingOptions,
) {
  await page.addInitScript(
    ({ apiKey: initApiKey }) => {
      window.localStorage.removeItem("openhands-onboarded");
      window.localStorage.setItem("analytics-consent", "false");
      window.localStorage.setItem("openhands-telemetry-consent", "denied");
      window.localStorage.setItem("openhands-telemetry-first-use", "true");
      window.localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "default-local",
            name: "Local",
            host: window.location.origin,
            apiKey: initApiKey,
            kind: "local",
          },
        ]),
      );
      window.localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "default-local", orgId: null }),
      );
    },
    { apiKey },
  );

  await beforeGoto?.();
  await routeOnboardingLlmCatalog(page);
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByTestId("onboarding-modal"),
    "onboarding modal should appear for first-run users",
  ).toBeVisible({ timeout: 20_000 });
}

export async function waitForOnboardingStep(page: Page, step: number) {
  await expect(
    page.getByTestId("onboarding-slide-rail"),
    `onboarding slide rail should show step ${step}`,
  ).toHaveAttribute("data-current-step", String(step), { timeout: 15_000 });
}

export async function waitForOnboardingBackendConnected(page: Page) {
  // The mock suites route the backend health probe to deterministic local/MSW
  // responses. Wait for the connected banner before advancing so the Next
  // button is enabled without relying on a fixed sleep.
  await expect(
    page.getByTestId("onboarding-backend-connected"),
    "onboarding backend health probe should report connected",
  ).toBeVisible({ timeout: 10_000 });
}

export async function waitForOnboardingLlmSettingsReady(page: Page) {
  await expect(
    page.locator('input[name="llm-provider-input"]'),
    "onboarding LLM provider input should be ready",
  ).toBeVisible({ timeout: 10_000 });
  await expect(
    page.locator('input[name="llm-model-input"]'),
    "onboarding LLM model input should be ready",
  ).toBeVisible({ timeout: 10_000 });
}

export async function clickOnboardingStepButton(page: Page, testId: string) {
  // Snapshot slides are translated and clipped during transitions. In CI,
  // Playwright can resolve the button as visible but outside the viewport.
  await page.getByTestId(testId).dispatchEvent("click");
}

export async function getOnboardingStepLayout(
  page: Page,
): Promise<OnboardingStepLayout> {
  await waitForOnboardingStep(page, ONBOARDING_BACKEND_STEP);

  // Wait for the backend health probe to settle before sampling the layout.
  // While the probe is in flight the backend slide shows a transient
  // "checking" banner at step 0. A healthy configured backend then auto-skips
  // the backend slide (starting the flow on agent selection), while an
  // unhealthy/unconfigured one keeps it. Sampling during the "checking" window
  // races that skip: we would wrongly report a backend step that immediately
  // disappears and then wait forever for a "connected" banner that never
  // renders. Waiting for "checking" to clear makes the verdict deterministic
  // regardless of how long the probe takes (e.g. when a spec slows
  // `/api/settings` via a `route.fetch()` intercept).
  await expect(
    page.getByTestId("onboarding-backend-checking"),
    "onboarding backend health probe should settle before sampling the layout",
  ).toHaveCount(0, { timeout: 15_000 });

  const hasBackendStep = await page
    .getByTestId("onboarding-step-check-backend")
    .isVisible()
    .catch(() => false);

  return hasBackendStep
    ? {
        hasBackendStep: true,
        agentStep: ONBOARDING_AGENT_STEP,
        llmStep: ONBOARDING_LLM_STEP,
        helloStep: ONBOARDING_HELLO_STEP,
      }
    : {
        hasBackendStep: false,
        agentStep: ONBOARDING_BACKEND_STEP,
        llmStep: ONBOARDING_AGENT_STEP,
        helloStep: ONBOARDING_LLM_STEP,
      };
}

export async function advanceOnboardingToLlmStep(page: Page) {
  const layout = await getOnboardingStepLayout(page);

  if (layout.hasBackendStep) {
    await waitForOnboardingBackendConnected(page);
    await clickOnboardingStepButton(page, "onboarding-backend-next");
    await waitForOnboardingStep(page, layout.agentStep);
  } else {
    await expect(
      page.getByTestId("onboarding-step-choose-agent"),
      "onboarding should start on agent selection when backend is healthy",
    ).toBeVisible({ timeout: 10_000 });
  }

  await clickOnboardingStepButton(page, "onboarding-agent-next");

  await waitForOnboardingStep(page, layout.llmStep);
  await expect(
    page.getByTestId("onboarding-step-setup-llm"),
    "onboarding LLM setup step should be visible after advancing",
  ).toBeVisible({ timeout: 10_000 });
}
