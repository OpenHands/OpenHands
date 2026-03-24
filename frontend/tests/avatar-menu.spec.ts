import test, { expect } from "@playwright/test";

/**
 * Test for issue #11933: Avatar context menu closes when moving cursor diagonally
 *
 * This test verifies that the user can move their cursor diagonally from the
 * avatar to the context menu without the menu closing unexpectedly.
 *
 * The component supports both CSS hover and click-to-toggle for the menu.
 * We use click-to-toggle which is more reliable in automated tests than
 * CSS hover simulation.
 */
test("avatar context menu stays open when moving cursor diagonally to menu", async ({
  page,
}) => {
  // Intercept GET /api/settings to return settings with a configured provider.
  // In OSS mode, the user context menu only renders when providers are configured.
  await page.route("**/api/settings", async (route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          llm_model: "openhands/claude-opus-4-5-20251101",
          llm_base_url: "",
          agent: "CodeActAgent",
          language: "en",
          llm_api_key: null,
          llm_api_key_set: false,
          search_api_key_set: false,
          confirmation_mode: false,
          security_analyzer: "llm",
          remote_runtime_resource_factor: 1,
          provider_tokens_set: { github: "" },
          enable_default_condenser: true,
          condenser_max_size: 240,
          enable_sound_notifications: false,
          user_consents_to_analytics: false,
          enable_proactive_conversation_starters: false,
          enable_solvability_analysis: false,
          max_budget_per_task: null,
        }),
      });
    } else {
      await route.continue();
    }
  });

  await page.goto("/");

  // Wait for the page to be fully loaded and check for AI config modal
  // The modal may appear for new users in OSS mode without settings
  const aiConfigModal = page.getByTestId("ai-config-modal");

  // Give the modal a chance to appear (it may load asynchronously)
  try {
    await aiConfigModal.waitFor({ state: "visible", timeout: 3000 });
    // Modal appeared - dismiss it by clicking save
    await page.getByTestId("save-settings-button").click();
    await expect(aiConfigModal).toBeHidden();
  } catch {
    // Modal didn't appear within timeout - that's fine, continue with test
  }

  const userAvatar = page.getByTestId("user-avatar");
  await expect(userAvatar).toBeVisible();

  // Use force:true to bypass the hover bridge pseudo-element that can
  // intercept clicks when the mouse triggers group-hover state
  await userAvatar.click({ force: true });

  // The context menu should appear via CSS group-hover
  const contextMenu = page.getByTestId("user-context-menu");
  await expect(contextMenu).toBeVisible();

  const menuWrapper = contextMenu.locator("..");
  await expect(menuWrapper).toHaveCSS("opacity", "1");

  // Now test diagonal mouse movement - move from avatar to menu
  // The menu should stay open due to the hover bridge in the component
  const avatarBox = await userAvatar.boundingBox();
  const menuBox = await contextMenu.boundingBox();

  if (!avatarBox || !menuBox) {
    throw new Error("Could not get bounding boxes");
  }

  // Move diagonally from avatar center toward the menu
  const startX = avatarBox.x + avatarBox.width / 2;
  const startY = avatarBox.y + avatarBox.height / 2;
  const endX = menuBox.x + menuBox.width / 2;
  const endY = menuBox.y + menuBox.height / 2;

  // Simulate diagonal movement with multiple steps
  const steps = 5;
  for (let i = 0; i <= steps; i++) {
    const x = startX + ((endX - startX) * i) / steps;
    const y = startY + ((endY - startY) * i) / steps;
    await page.mouse.move(x, y);
  }

  // Menu should still be visible after diagonal movement
  await expect(contextMenu).toBeVisible();
  await expect(menuWrapper).toHaveCSS("opacity", "1");
});
