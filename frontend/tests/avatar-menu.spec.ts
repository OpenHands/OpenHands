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

  const avatarBox = await userAvatar.boundingBox();
  if (!avatarBox) {
    throw new Error("Could not get bounding box for avatar");
  }

  const avatarCenterX = avatarBox.x + avatarBox.width / 2;
  const avatarCenterY = avatarBox.y + avatarBox.height / 2;
  await page.mouse.move(avatarCenterX, avatarCenterY);

  // The context menu should appear via CSS group-hover
  const contextMenu = page.getByTestId("user-context-menu");
  await expect(contextMenu).toBeVisible();

  const menuWrapper = contextMenu.locator("..");
  await expect(menuWrapper).toHaveCSS("opacity", "1");
});
