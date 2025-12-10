import test, { expect } from "@playwright/test";

/**
 * Test for issue #11933: Avatar context menu closes when moving cursor diagonally
 *
 * This test verifies that the user can move their cursor diagonally from the
 * avatar to the context menu without the menu closing unexpectedly.
 */
test("avatar context menu stays open when moving cursor diagonally to menu", async ({
  page,
}) => {
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

  // Wait for hover state to update
  await page.waitForTimeout(100);

  // The menu uses opacity-0/opacity-100 for visibility, so we need to check CSS
  // not just DOM visibility. When not hovered, it has opacity-0 and pointer-events-none
  const opacity = await contextMenu.evaluate(
    (el) => window.getComputedStyle(el.parentElement!).opacity,
  );

  // The menu should still be interactive (opacity 1) to allow diagonal access
  // This assertion will FAIL - menu becomes opacity-0 when leaving the avatar area
  expect(opacity).toBe("1");
});
