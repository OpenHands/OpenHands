import { test, expect, type Page } from "@playwright/test";
import {
  seedLocalStorage,
  routeSessionApiKey,
  ensureMockLLMProfile,
  setChatInput,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  waitForNonUserMessageText,
  deleteConversation,
} from "../utils/mock-llm-helpers";

const FILLED_REPLY = "AUTOMATION_FORM_FILLED_OK";
const OTHER_REPLY = "OTHER_CONVERSATION_OK";
const SETUP_READY_REPLY = "AUTOMATION_SETUP_READY_OK";
const AGENT_NAME = "Nightly Issue Triage";
const AGENT_PROMPT = "Triage new issues every night and label them";

function conversationIdFromUrl(page: Page) {
  return new URL(page.url()).pathname.split("/").pop() ?? "";
}

async function waitForStartedConversation(page: Page) {
  await expect(page).toHaveURL(/\/conversations\/(?!launch-)[^/?#]+/, {
    timeout: 45_000,
  });
  return conversationIdFromUrl(page);
}

// Real Canvas, automation API and agent-server; only inference is scripted.
// The agent fills the setup form with its form tool, then the user opens
// another conversation from the sidebar and comes back.
test("automation setup form keeps agent-filled fields after switching conversations", async ({
  page,
  request,
}, testInfo) => {
  test.setTimeout(180_000);
  await seedLocalStorage(page);
  await routeSessionApiKey(page);
  await ensureMockLLMProfile(page);
  await resetMockLLM(request);
  const conversationIds: string[] = [];
  const capture = async (name: string) => {
    const path = testInfo.outputPath(`${name}.png`);
    await page.screenshot({ path });
    await testInfo.attach(name, { path, contentType: "image/png" });
  };

  try {
    // An ordinary conversation to switch to.
    await registerTrajectory(request, "form-return-other", [
      { text: "" },
      { text: OTHER_REPLY },
    ]);
    await activateTrajectory(request, "form-return-other");
    await page.goto("/");
    await setChatInput(page, "Say hello");
    await page.getByTestId("submit-button").click();
    const otherId = await waitForStartedConversation(page);
    conversationIds.push(otherId);
    await waitForNonUserMessageText(page, OTHER_REPLY, 45_000);

    // The setup conversation, where the agent fills the form.
    await registerTrajectory(request, "form-return-setup", [
      { text: "" },
      { text: SETUP_READY_REPLY },
      {
        tool_call: {
          name: "automation_form_update",
          arguments: {
            fields: {
              name: AGENT_NAME,
              prompt: AGENT_PROMPT,
              frequency: "weekly",
            },
          },
        },
      },
      { text: FILLED_REPLY },
    ]);
    await activateTrajectory(request, "form-return-setup");
    await page.goto("/automations/setup");
    await setChatInput(page, "Triage my issues every week");
    await page.getByTestId("submit-button").click();
    const setupId = await waitForStartedConversation(page);
    conversationIds.push(setupId);
    await waitForNonUserMessageText(page, SETUP_READY_REPLY, 45_000);
    // Fill the form from a later turn, while the conversation is open.
    await setChatInput(page, "Make it weekly and name it");
    await page.getByTestId("submit-button").click();
    await waitForNonUserMessageText(page, FILLED_REPLY, 45_000);

    const nameField = page.getByTestId("automation-setup-name");
    const promptField = page.getByTestId("automation-setup-prompt");
    const weekly = page.getByTestId("automation-setup-frequency-weekly");
    await expect(promptField).toHaveValue(AGENT_PROMPT, { timeout: 15_000 });
    await expect(nameField).toHaveValue(AGENT_NAME);
    await expect(weekly).toHaveAttribute("aria-checked", "true");
    // The user's own edit on top of the agent's.
    await nameField.click();
    await nameField.press("End");
    await nameField.pressSequentially(" v2");
    await expect(nameField).toHaveValue(`${AGENT_NAME} v2`);
    await capture("1-agent-filled");

    // Switch conversations in the app, without reloading the page.
    const conversationLink = (id: string) =>
      page
        .getByTestId("conversation-panel")
        .locator(`a[href*="/conversations/${id}"]`)
        .first();
    await conversationLink(otherId).click();
    await expect(page).toHaveURL(
      new RegExp(`/conversations/${otherId}(\\?|$)`),
    );
    await expect(promptField).toHaveCount(0);
    await capture("2-other-conversation");

    await conversationLink(setupId).click();
    await expect(page).toHaveURL(
      new RegExp(`/conversations/${setupId}(\\?|$)`),
    );
    await expect(promptField).toBeVisible({ timeout: 15_000 });
    // Read the form as it first appears; replaying the agent's update later
    // would not count as keeping it.
    const returned = {
      prompt: await promptField.inputValue(),
      name: await nameField.inputValue(),
      weekly: await weekly.getAttribute("aria-checked"),
    };
    await capture("3-returned");
    expect(returned).toEqual({
      prompt: AGENT_PROMPT,
      name: `${AGENT_NAME} v2`,
      weekly: "true",
    });
  } finally {
    for (const id of conversationIds) await deleteConversation(request, id);
    await resetMockLLM(request);
  }
});
