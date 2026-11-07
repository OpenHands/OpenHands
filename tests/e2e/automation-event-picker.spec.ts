import { expect, test, type Locator } from "@playwright/test";

test.use({
  video: { mode: "on", size: { width: 1440, height: 1000 } },
  viewport: { width: 1440, height: 1000 },
});

async function typeInto(field: Locator, value: string) {
  await field.click();
  await field.press("ControlOrMeta+A");
  await field.press("Backspace");
  if (value) await field.pressSequentially(value, { delay: 35 });
}

test("edits multiple event keys without changing the shared filter", async ({
  page,
}, testInfo) => {
  await page.addInitScript(() => {
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "default-local",
          name: "Local",
          host: location.origin,
          apiKey: "",
          kind: "local",
        },
      ]),
    );
    localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "default-local", orgId: null }),
    );
  });
  await page.goto("/automations/setup");
  await page
    .getByRole("checkbox", { name: /Send anonymous usage data/ })
    .uncheck();
  await page.getByTestId("confirm-telemetry-preferences").click();
  await expect(page.getByTestId("telemetry-consent-form")).toHaveCount(0);
  await typeInto(
    page.getByTestId("automation-setup-name"),
    "Review pull request updates",
  );
  await typeInto(
    page.getByTestId("automation-setup-prompt"),
    "Review the pull request and summarize actionable findings.",
  );
  await page.getByRole("radio", { name: /^Event/ }).click();
  const picker = page.getByTestId("automation-event-picker");
  const search = page.getByTestId("automation-setup-event-key");
  await picker.getByRole("button", { name: /^Remove / }).click();
  await typeInto(search, "Pull request");
  await expect(
    page.getByRole("option", { name: /Pull request opened/ }),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("01-event-search.png") });
  await page.getByRole("option", { name: /Pull request opened/ }).click();
  await typeInto(search, "commits updated");
  await page
    .getByRole("option", { name: /Pull request commits updated/ })
    .click();
  await search.press("Escape");
  await expect(
    picker.getByRole("button", { name: "Remove Pull request opened" }),
  ).toBeVisible();
  await expect(
    picker.getByRole("button", { name: "Remove Pull request commits updated" }),
  ).toBeVisible();
  await page.getByTestId("automation-setup-add-event-filter").click();
  const filter = page.getByTestId("automation-setup-event-filter");
  await typeInto(filter, "repository.full_name == 'acme/app'");
  await typeInto(search, "custom:*");
  await search.press("Escape");
  await typeInto(search, "");
  await search.press("Escape");
  await page.screenshot({
    path: testInfo.outputPath("02-selected-events-and-filter.png"),
  });
  const savedResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/v1/drafts") &&
      response.request().method() === "POST",
  );
  const saved = page.waitForRequest(
    (request) =>
      request.url().endsWith("/v1/drafts") && request.method() === "POST",
  );
  await page.getByRole("button", { name: "Save draft", exact: true }).click();
  expect((await savedResponse).status()).toBe(201);
  expect((await saved).postDataJSON().draft.trigger).toEqual({
    type: "event",
    source: "github",
    on: ["pull_request.opened", "pull_request.synchronize"],
    filter: "repository.full_name == 'acme/app'",
  });
  await expect(
    page.getByRole("button", { name: "Save draft", exact: true }),
  ).toBeEnabled();
  await page.setViewportSize({ width: 390, height: 844 });
  await typeInto(search, "Pull request");
  await expect(
    page.getByRole("option", { name: /Pull request opened/ }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("03-phone-event-picker.png"),
  });
});
