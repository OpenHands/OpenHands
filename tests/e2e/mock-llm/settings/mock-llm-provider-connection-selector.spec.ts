import { test, expect } from "@playwright/test";
import {
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
} from "../utils/mock-llm-helpers";

test.describe.configure({ mode: "serial" });

test.describe("provider connection selector", () => {
  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test("selects a supported provider in Add provider", async ({ page }) => {
    test.setTimeout(120_000);

    await routeSessionApiKey(page);
    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "add-provider-connection");

    await page.getByTestId("add-provider-connection").click();

    const modal = page.getByTestId("provider-connection-modal");
    await expect(modal).toBeVisible();

    const providerSelector = modal.getByRole("combobox", {
      name: /provider/i,
    });
    await expect(providerSelector).toBeVisible();

    await providerSelector.click();

    const openAI = page.getByTestId("provider-item-openai");
    await expect(openAI).toBeVisible();

    // Leave the dropdown open long enough for the recorded animation
    // to clearly demonstrate supported-provider choices.
    await page.waitForTimeout(1_200);

    await openAI.click();
    await expect(providerSelector).toHaveValue("OpenAI");

    // Hold the selected state for the final frames of the evidence.
    await page.waitForTimeout(1_200);
  });
});
