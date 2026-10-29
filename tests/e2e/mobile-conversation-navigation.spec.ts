import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

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
  const topBarBounds = await topBarMenu.boundingBox();
  expect(topBarBounds!.width).toBeGreaterThanOrEqual(48);
  expect(topBarBounds!.height).toBeGreaterThanOrEqual(48);
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
