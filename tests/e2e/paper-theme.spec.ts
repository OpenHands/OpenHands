import { expect, test, type Locator, type Page } from "@playwright/test";

test.beforeEach(async ({ page }) => {
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

async function dismissMockConsent(page: Page) {
  const consent = page.getByRole("dialog", { name: "Help improve OpenHands" });
  await expect(consent).toBeVisible({ timeout: 30000 });
  await consent.getByRole("checkbox").uncheck();
  await consent.getByRole("button", { name: "Confirm preferences" }).click();
  await expect(consent).toBeHidden();
}

async function openMockPage(page: Page, route: string) {
  await page.goto(route);
  await dismissMockConsent(page);
}

async function selectTheme(page: Page, name: string) {
  const input = page.getByRole("combobox", { name: /color theme/i });
  await input.scrollIntoViewIfNeeded();
  await expect(input).toBeInViewport();
  await input.fill(name);
  await expect(page.getByRole("option", { name, exact: true })).toBeVisible();
  await page.getByRole("option", { name, exact: true }).click();
}

async function expectFitsViewport(element: Locator, width: number) {
  await expect(element).toBeVisible();
  const bounds = (await element.boundingBox())!;
  expect(bounds.x).toBeGreaterThanOrEqual(-1);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(width + 1);
}

async function textContrast(element: Locator) {
  return element.evaluate((node) => {
    const luminance = (color: string) => {
      const channels = color
        .match(/[\d.]+/g)!
        .slice(0, 3)
        .map(Number);
      const linear = channels.map((value) => {
        const channel = value / 255;
        return channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    };
    let surface: Element | null = node;
    let background = "rgba(0, 0, 0, 0)";
    while (surface && background === "rgba(0, 0, 0, 0)") {
      background = getComputedStyle(surface).backgroundColor;
      surface = surface.parentElement;
    }
    const ink = luminance(getComputedStyle(node).color);
    const paper = luminance(background);
    return (Math.max(ink, paper) + 0.05) / (Math.min(ink, paper) + 0.05);
  });
}

test("Paper selection persists and returning to Neutral restores typography and color", async ({
  page,
}) => {
  // Arrange: use the actual settings selector, without saving other settings.
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openMockPage(page, "/settings/app");
  const heading = page.locator("main").last().getByRole("heading").first();
  const themeInput = page.getByRole("combobox", { name: /color theme/i });
  await expect(themeInput).toHaveValue("OpenHands-Neutral");
  const originalFont = await heading.evaluate(
    (el) => getComputedStyle(el).fontFamily,
  );
  const originalBackground = await page
    .locator("body")
    .evaluate((el) => getComputedStyle(el).backgroundColor);

  // Act: Paper applies immediately and survives a fresh app initialization.
  await selectTheme(page, "Paper");
  await expect(heading).toHaveCSS("font-family", /Canvas Newsreader/);
  await page.reload();
  await dismissMockConsent(page);

  // Assert the rendered contract, including the bundled face and dense controls.
  await expect(themeInput).toHaveValue("Paper");
  await expect(page.locator("[data-color-theme]").first()).toHaveAttribute(
    "data-color-scheme",
    "light",
  );
  await page.evaluate(() => document.fonts.ready);
  expect(
    await page.evaluate(() =>
      document.fonts.check('500 20px "Canvas Newsreader"'),
    ),
  ).toBe(true);
  await expect(heading).toHaveCSS("font-family", /Canvas Newsreader/);
  await expect(themeInput).not.toHaveCSS("font-family", /Newsreader/);
  await expectFitsViewport(page.getByTestId("app-settings-screen"), 1440);
  expect(await textContrast(themeInput)).toBeGreaterThanOrEqual(4.5);
  const save = page.getByTestId("submit-button");
  await expect(save).toBeDisabled();
  await page.getByTestId("git-user-name-input").fill("Theme preview");
  await expect(save).toBeEnabled();
  // Enabling snaps the foreground immediately while its background settles
  // through the existing control transition; inspect the completed state.
  await save.evaluate(async (el) => {
    await Promise.all(
      el.getAnimations().map((animation) => animation.finished),
    );
  });
  expect(await textContrast(save)).toBeGreaterThanOrEqual(4.5);

  await selectTheme(page, "OpenHands-Neutral");
  await expect(heading).toHaveCSS("font-family", originalFont);
  await expect(page.locator("body")).toHaveCSS(
    "background-color",
    originalBackground,
  );
  await expect(page.locator("[data-color-theme]").first()).toHaveAttribute(
    "data-color-scheme",
    "dark",
  );
});

test("Paper remains usable at phone width with keyboard navigation and reduced motion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() =>
    localStorage.setItem("openhands-color-theme", "paper"),
  );
  await openMockPage(page, "/settings/app");
  const themeInput = page.getByRole("combobox", { name: /color theme/i });
  await expectFitsViewport(page.getByTestId("app-settings-screen"), 390);
  await expectFitsViewport(themeInput, 390);

  // Focus via the keyboard and inspect the portalled list without selecting
  // another theme. No backend or model execution is involved.
  await themeInput.focus();
  await page.keyboard.press("ArrowDown");
  const paperOption = page.getByRole("option", { name: "Paper", exact: true });
  await expectFitsViewport(page.getByRole("listbox"), 390);
  await expect(paperOption).toBeVisible();
  expect(await themeInput.evaluate((el) => el.matches(":focus-visible"))).toBe(
    true,
  );
  await page.keyboard.press("Escape");
  await expect(page.getByRole("listbox")).toBeHidden();

  await openMockPage(page, "/conversations/canvas-demo");
  const message = page.getByTestId("agent-message").last();
  await expect(message).toContainText("I generated the project canvas.");
  await expect(message).not.toHaveCSS("font-family", /Newsreader/);
  await expectFitsViewport(page.getByTestId("chat-interface"), 390);
  const menu = page.getByTestId("sidebar-mobile-menu-toggle");
  await expect(menu).toHaveCSS("transition-property", "none");
  await menu.focus();
  await page.keyboard.press("Enter");
  const drawer = page.getByTestId("sidebar-mobile-drawer");
  await expectFitsViewport(drawer, 390);
  await page.keyboard.press("Escape");
  await expect(drawer).toBeHidden();
});
