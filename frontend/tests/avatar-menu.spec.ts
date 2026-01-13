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
  // Skip on WebKit - Playwright's mouse.move() doesn't reliably trigger CSS hover states
  test.skip(browserName === "webkit", "Playwright hover simulation unreliable");

  await page.goto("/");

  // Get the user avatar button
  const userAvatar = page.getByTestId("user-avatar");
  await expect(userAvatar).toBeVisible();

  // Get avatar bounding box first
  const avatarBox = await userAvatar.boundingBox();
  if (!avatarBox) {
    throw new Error("Could not get bounding box for avatar");
  }

  // Use mouse.move to hover (not .hover() which may trigger click)
  const avatarCenterX = avatarBox.x + avatarBox.width / 2;
  const avatarCenterY = avatarBox.y + avatarBox.height / 2;
  await page.mouse.move(avatarCenterX, avatarCenterY);

  // The context menu should appear via CSS group-hover
  const contextMenu = page.getByTestId("account-settings-context-menu");
  await expect(contextMenu).toBeVisible();

  // Move UP from the LEFT side of the avatar - simulating diagonal movement
  // toward the menu (which is to the right). This exits the hover zone.
  const leftX = avatarBox.x + 2;
  const aboveY = avatarBox.y - 50;
  await page.mouse.move(leftX, aboveY);

  // The menu uses opacity-0/opacity-100 for visibility via CSS.
  // Use toHaveCSS which auto-retries, avoiding flaky waitForTimeout.
  // The menu should remain visible (opacity 1) to allow diagonal access to it.
  const menuWrapper = contextMenu.locator("..");
  await expect(menuWrapper).toHaveCSS("opacity", "1");
});
