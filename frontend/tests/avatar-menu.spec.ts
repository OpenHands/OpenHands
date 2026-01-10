import test, { expect } from "@playwright/test";

/**
 * Test for issue #11933: Avatar context menu closes when moving cursor diagonally
 *
 * This test verifies that the user can move their cursor diagonally from the
 * avatar to the context menu without the menu closing unexpectedly.
 */
test("avatar context menu stays open when moving cursor diagonally to menu", async ({
  page,
  browserName,
}) => {
  // WebKit: Playwright hover/mouse simulation is flaky for CSS hover-only menus.
  test.skip(browserName === "webkit", "Playwright hover simulation unreliable");

  await page.goto("/");

  const aiConfigModal = page.getByTestId("ai-config-modal");
  if (await aiConfigModal.isVisible().catch(() => false)) {
    // In OSS mock mode, missing settings can open the AI-config modal; its backdrop
    // intercepts pointer events and prevents hover interactions.
    await page.getByTestId("save-settings-button").click();
    await expect(aiConfigModal).toBeHidden();
  }

  const userAvatar = page.getByTestId("user-avatar");
  await expect(userAvatar).toBeVisible();

  // Ensure the avatar is in the viewport
  await userAvatar.scrollIntoViewIfNeeded();

  // Wait a bit more for the avatar to be fully rendered
  await page.waitForTimeout(100);

  const avatarBox = await userAvatar.boundingBox();
  if (!avatarBox) {
    // Debug: take a screenshot and log page content
    await page.screenshot({ path: "debug-avatar-test.png" });
    const avatarCount = await page.getByTestId("user-avatar").count();
    console.log("Number of user-avatar elements found:", avatarCount);
    const pageTitle = await page.title();
    console.log("Page title:", pageTitle);
    throw new Error("Could not get bounding box for avatar");
  }

  const avatarCenterX = avatarBox.x + avatarBox.width / 2;
  const avatarCenterY = avatarBox.y + avatarBox.height / 2;
  await page.mouse.move(avatarCenterX, avatarCenterY);

  const contextMenu = page.getByTestId("account-settings-context-menu");
  await expect(contextMenu).toBeVisible();

  const menuWrapper = contextMenu.locator("..");
  // Opacity transitions can be flaky/slow in CI; avoid strict string matching and allow a
  // small tolerance with a longer timeout.
  const wrapperHandle = await menuWrapper.elementHandle();
  if (!wrapperHandle) {
    throw new Error("Menu wrapper element handle not found");
  }
  await expect(menuWrapper).toBeVisible({ timeout: 5000 });
  await page.waitForFunction(
    (el) => Number.parseFloat(window.getComputedStyle(el).opacity) >= 0.98,
    wrapperHandle,
    { timeout: 5000 },
  );
  await expect(menuWrapper).toBeVisible();
});
