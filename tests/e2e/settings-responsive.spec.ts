import { expect, test, type Page } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem(
      "openhands-sidebar",
      JSON.stringify({ state: { collapsed: false }, version: 0 }),
    );
    localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "default-local",
          name: "Local QA",
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
});

async function openMockPage(page: Page, route: string) {
  await page.goto(route);
  const consent = page.getByRole("dialog", { name: "Help improve OpenHands" });
  await expect(consent).toBeVisible({ timeout: 30000 });
  await consent.getByRole("checkbox").uncheck();
  await consent.getByRole("button", { name: "Confirm preferences" }).click();
  await expect(consent).toBeHidden();
}

for (const width of [390, 767, 768, 820, 1023, 1024, 1440]) {
  test(`settings and customization remain usable at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1024 });
    await openMockPage(page, "/customize");
    const compact = width < 1024;
    if (compact) {
      await expect(page.getByTestId("extensions-mobile-hub")).toBeVisible();
      // Keyboard navigation out of the hub remains available.
      await page
        .getByTestId("extensions-mobile-hub")
        .getByRole("link", { name: "MCP Servers", exact: true })
        .press("Enter");
    }
    await expect(page.getByTestId("mcp-add-custom-server")).toBeVisible();
    const nav = page.getByTestId("extensions-navbar-desktop");
    if (compact) await expect(nav).toBeHidden();
    else await expect(nav).toBeVisible();
    const search = page.getByPlaceholder("Search MCP servers");
    await expect(search).toBeVisible();
    expect((await search.boundingBox())!.width).toBeGreaterThan(150);
    await search.fill("GitHub");
    await expect(
      page.getByRole("heading", { name: "GitHub", exact: true }),
    ).toBeVisible();
    await page.getByTestId("mcp-add-custom-server").click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.keyboard.press("Escape");

    for (const route of ["/skills", "/plugins", "/apps", "/settings/app"]) {
      await openMockPage(page, route);
      const main = page.locator("main").last();
      await expect(main).toBeVisible();
      expect((await main.boundingBox())!.width).toBeGreaterThan(340);
      expect(
        await main.evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
      ).toBe(true);
    }
    await openMockPage(page, "/settings");
    if (compact)
      await expect(page.getByTestId("settings-mobile-hub")).toBeVisible();
    else
      await expect(page.getByTestId("settings-navbar-desktop")).toBeVisible();
  });
}
