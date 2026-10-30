import { expect, test, type Locator, type Page } from "@playwright/test";

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
    if (compact) {
      // As on Customize, Settings' main fills the column and scrolls itself,
      // so a popover inside it is not clipped on a short page (#18255).
      for (const [route, scrolls] of [
        ["/settings/secrets", false],
        ["/settings/app", true],
      ] as const) {
        await openMockPage(page, route);
        const main = page.locator("main").last();
        await expect(main).toBeVisible();
        const fit = await main.evaluate((el) => ({
          bottom: el.getBoundingClientRect().bottom,
          scrolls: el.scrollHeight > el.clientHeight,
        }));
        expect(Math.abs(fit.bottom - 1024)).toBeLessThanOrEqual(1);
        expect(fit.scrolls).toBe(scrolls);
      }
    }
    await openMockPage(page, "/settings");
    if (compact)
      await expect(page.getByTestId("settings-mobile-hub")).toBeVisible();
    else
      await expect(page.getByTestId("settings-navbar-desktop")).toBeVisible();
  });
}

test("switch thumbs stay inside their tracks in a right-to-left language", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1024 });
  await openMockPage(page, "/settings/app");
  await page.getByTestId("language-input").click();
  await page.getByRole("option", { name: "Arabic" }).click();
  await page.getByTestId("submit-button").click();
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");

  // Thumbs slide for 200ms after a toggle or a direction change, so wait
  // for the slide to finish before measuring.
  const thumbOverflow = (track: Locator) =>
    track.evaluate(async (element) => {
      const thumb = element.firstElementChild!;
      await Promise.all(thumb.getAnimations().map((slide) => slide.finished));
      const outer = element.getBoundingClientRect();
      const inner = thumb.getBoundingClientRect();
      return Math.max(outer.left - inner.left, inner.right - outer.right);
    });

  // Settings switches (`md`). The checklist switch starts on and the sound
  // switch off, so both thumb positions are covered.
  for (const [testId, checked] of [
    ["show-getting-started-checklist-switch", true],
    ["enable-sound-notifications-switch", false],
  ] as const) {
    const input = page.getByTestId(testId);
    if (checked) await expect(input).toBeChecked();
    else await expect(input).not.toBeChecked();
    const track = page
      .locator("label", { has: input })
      .locator("span[aria-hidden='true']");
    expect(await thumbOverflow(track)).toBeLessThanOrEqual(0);
  }

  // Menu-row switch (`sm`): hide older conversations, off and then on.
  await page.getByTestId("conversation-layouts-toggle").click();
  await page.getByTestId("advanced-options-row").click();
  const hideOlder = page.getByTestId("toggle-older-conversations");
  const compactTrack = hideOlder.locator(":scope > span[aria-hidden='true']");
  for (const checked of ["false", "true"]) {
    if (checked === "true") await compactTrack.click();
    await expect(hideOlder).toHaveAttribute("aria-checked", checked);
    expect(await thumbOverflow(compactTrack)).toBeLessThanOrEqual(0);
  }
});
