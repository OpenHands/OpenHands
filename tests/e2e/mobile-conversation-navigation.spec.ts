import { expect, test, type CDPSession, type Page } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

type TouchPoint = [x: number, y: number];

/** Drives a one-finger swipe through Chromium's native touch input. */
async function swipe(cdp: CDPSession, from: TouchPoint, to: TouchPoint) {
  const steps = 8;
  const point = (step: number) => ({
    x: from[0] + ((to[0] - from[0]) * step) / steps,
    y: from[1] + ((to[1] - from[1]) * step) / steps,
  });
  await cdp.send("Input.dispatchTouchEvent", {
    type: "touchStart",
    touchPoints: [point(0)],
  });
  for (let step = 1; step <= steps; step += 1) {
    await cdp.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [point(step)],
    });
  }
  await cdp.send("Input.dispatchTouchEvent", {
    type: "touchEnd",
    touchPoints: [],
  });
}

const nextFrames = (page: Page) =>
  page.evaluate(
    () =>
      new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      }),
  );

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
  await page.goto("/conversations/1");
  const consent = page.getByRole("dialog", { name: "Help improve OpenHands" });
  await expect(consent).toBeVisible({ timeout: 30000 });
  await consent.getByRole("checkbox").uncheck();
  await consent.getByRole("button", { name: "Confirm preferences" }).click();
  await expect(consent).toBeHidden();
});

test("mobile conversation menu has a comfortable touch target", async ({
  page,
}) => {
  const menu = page.getByTestId("sidebar-mobile-menu-toggle");
  await expect(menu).toBeVisible();
  await expect(menu).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
  expect(
    (await menu.locator("svg").boundingBox())!.width,
  ).toBeGreaterThanOrEqual(24);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    const bounds = await menu.boundingBox();
    expect(bounds!.width).toBeGreaterThanOrEqual(48);
    expect(bounds!.height).toBeGreaterThanOrEqual(48);
    const header = await page.getByTestId("chat-pane-header").boundingBox();
    expect(bounds!.y).toBeGreaterThanOrEqual(header!.y);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(
      header!.y + header!.height,
    );
  }
  await menu.tap({ position: { x: 2, y: 2 } });
  const drawer = page.getByTestId("sidebar-mobile-drawer");
  await expect(drawer).toBeVisible();
  await drawer.getByTestId("sidebar-conversations-link").tap();
  await expect(drawer).toBeHidden();
  const topBarMenu = page.getByTestId("sidebar-mobile-menu-toggle");
  await expect(topBarMenu).toBeVisible();
  await expect(topBarMenu).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
  const topBarBounds = await topBarMenu.boundingBox();
  expect(topBarBounds!.width).toBeGreaterThanOrEqual(48);
  expect(topBarBounds!.height).toBeGreaterThanOrEqual(48);
});

test("swiping right from the chat's left edge opens the navigation drawer", async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Uses Chromium touch input.");
  const cdp = await page.context().newCDPSession(page);
  const menu = page.getByTestId("sidebar-mobile-menu-toggle");
  const drawer = page.getByTestId("sidebar-mobile-drawer");
  const chat = (await page
    .getByTestId("conversation-chat-panel")
    .boundingBox())!;
  const y = chat.y + chat.height / 2;

  await swipe(cdp, [chat.x + 6, y], [chat.x + 12, y - 240]);
  await swipe(cdp, [chat.x + 120, y], [chat.x + 320, y]);
  await swipe(cdp, [chat.x + 6, y], [chat.x + 40, y]);
  await nextFrames(page);
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await expect(drawer).toHaveCount(0);

  await swipe(cdp, [chat.x + 6, y], [chat.x + 220, y + 8]);
  await expect(drawer).toBeVisible();
  // The swipe opens the drawer rather than navigating the browser back.
  await expect(page).toHaveURL(/\/conversations\/1$/);
  await drawer.getByTestId("sidebar-mobile-drawer-close").tap();
  await expect(drawer).toBeHidden();

  await menu.tap();
  await expect(drawer).toBeVisible();
  await cdp.detach();
});

test("mobile conversation rows are easy to tap without triggering their actions", async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 844 });
  await page.getByTestId("sidebar-mobile-menu-toggle").tap();
  const drawer = page.getByTestId("sidebar-mobile-drawer");
  const row = drawer
    .getByTestId("conversation-card")
    .filter({ hasText: "Repo Testing" });
  await expect(row).toBeVisible();
  const bounds = await row.boundingBox();
  expect(bounds!.height).toBeGreaterThanOrEqual(48);
  expect((await row.getByTestId("ellipsis-button").boundingBox())!.width).toBe(
    24,
  );
  expect(
    (await row.getByRole("button", { name: "Pin conversation" }).boundingBox())!
      .width,
  ).toBe(22);
  await row.getByTestId("ellipsis-button").tap();
  await expect(page.getByTestId("context-menu")).toBeVisible();
  await expect(page).toHaveURL(/\/conversations\/1/);
  await page.keyboard.press("Escape");
  await row.getByRole("button", { name: "Pin conversation" }).tap();
  await expect(page).toHaveURL(/\/conversations\/1/);
  await row.tap({ position: { x: 10, y: bounds!.height - 3 } });
  await expect(page).toHaveURL(/\/conversations\/2/);
  await expect(drawer).toBeHidden();
});

test("desktop sidebar rows retain their compact dimensions", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 844 });
  const row = page
    .getByTestId("conversation-card")
    .filter({ hasText: "Repo Testing" });
  await expect(row).toBeVisible();
  expect((await row.boundingBox())!.height).toBe(32);
  expect(
    (await page.getByTestId("sidebar-conversations-link").boundingBox())!
      .height,
  ).toBe(36);
});

test("shared conversation keeps its header inside mobile safe areas", async ({
  page,
}) => {
  const conversationId = "mobile-safe-area";
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setSafeAreaInsetsOverride", {
    insets: { top: 44, right: 0, bottom: 34, left: 0 },
  });
  await page.goto(`/shared/conversations/${conversationId}`);

  const heading = page.getByRole("heading", {
    name: "Shared mobile conversation",
  });
  await expect(heading).toBeVisible();
  const headerBounds = await heading.boundingBox();
  const returnLinkBounds = await page
    .getByRole("link", { name: "OpenHands" })
    .boundingBox();
  expect(headerBounds!.y).toBeGreaterThanOrEqual(44);
  expect(returnLinkBounds!.y).toBeGreaterThanOrEqual(44);
  const shell = page.locator(".app-shell-safe-area");
  await expect(shell).toHaveCSS("padding-top", "44px");
  await expect(shell).toHaveCSS("padding-bottom", "34px");
  expect(
    await page.evaluate(() =>
      Math.max(
        document.documentElement.scrollHeight,
        document.body.scrollHeight,
      ),
    ),
  ).toBe(844);
  await cdp.detach();
});
