import { expect, test } from "@playwright/test";

// Regression for #17902: in a real browser React removes a clicked menu
// option before the document click listener runs, so the Filters popover
// used to treat every option click as an outside click and close.
test("keeps the automations Filters popover open while combining filters", async ({
  page,
}) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("analytics-consent", "false");
    window.localStorage.setItem("openhands-telemetry-consent", "denied");
    window.localStorage.setItem("openhands-telemetry-first-use", "true");
    window.localStorage.setItem("openhands-onboarded", "1");
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: window.location.origin,
          apiKey: "",
          kind: "local",
        },
      ]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "default-local", orgId: null }),
    );
  });
  await page.goto("/automations");
  // Mock settings have no analytics choice yet, so the local consent prompt
  // covers the page until it is answered.
  await page.getByTestId("confirm-telemetry-preferences").click();
  await expect(page.getByTestId("telemetry-consent-form")).toHaveCount(0);

  const filtersTrigger = page
    .getByTestId("automations-filters")
    .getByTestId("dropdown-trigger")
    .first();
  const filtersMenu = page.getByTestId("automations-filters-menu");

  await filtersTrigger.click();
  await page
    .getByTestId("automations-filter-status")
    .getByTestId("dropdown-trigger")
    .click();
  await page.getByTestId("automations-filter-status-disabled").click();

  await expect(filtersMenu).toBeVisible();
  await expect(filtersTrigger).toHaveAttribute("aria-expanded", "true");

  await page
    .getByTestId("automations-filter-trigger")
    .getByTestId("dropdown-trigger")
    .click();
  await page.getByTestId("automations-filter-trigger-schedule").click();

  await expect(filtersMenu).toBeVisible();
  await expect(filtersTrigger).toContainText("2");

  await page.getByTestId("automations-filters-reset").click();

  await expect(filtersMenu).toBeVisible();
  await expect(page.getByTestId("automations-filters-reset")).toHaveCount(0);

  await page.getByRole("heading", { name: "Dashboard" }).click();

  await expect(filtersMenu).toHaveCount(0);
  await expect(filtersTrigger).toHaveAttribute("aria-expanded", "false");
});
