import { test, expect } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  ensureMockLLMProfile,
  setChatInput,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  getMockLLMRequests,
  waitForNonUserMessageText,
  deleteConversation,
} from "../utils/mock-llm-helpers";

// Real Canvas, automation API and agent-server; only inference is scripted.
// Hold the create request to prove navigation does not depend on backend speed.
test("automation help opens immediately and sends the selected, unsaved form to the model", async ({
  page,
  request,
}, testInfo) => {
  await seedLocalStorage(page);
  await routeSessionApiKey(page);
  await ensureMockLLMProfile(page);
  await resetMockLLM(request);
  await registerTrajectory(request, "automation-help", [
    // Conversation title generation and the agent turn share the model.
    { text: "AUTOMATION_HELP_CONTEXT_RECEIVED" },
    { text: "AUTOMATION_HELP_CONTEXT_RECEIVED" },
  ]);
  await activateTrajectory(request, "automation-help");
  const headers = { "X-Session-API-Key": SESSION_API_KEY };
  const created = await request.post(
    `${BACKEND_URL}/api/automation/v1/preset/prompt`,
    {
      headers,
      data: {
        name: "OSS PR Reviewer",
        prompt: "Review requested pull requests",
        enabled: false,
        trigger: { type: "cron", schedule: "0 9 * * *", timezone: "UTC" },
      },
    },
  );
  expect(created.ok(), "create isolated automation fixture").toBe(true);
  const automationId = (await created.json()).id as string;
  let conversationId: string | undefined;
  let rejectNextCreate = true;
  let releaseCreate!: () => void;
  const createGate = new Promise<void>((resolve) => {
    releaseCreate = resolve;
  });
  await page.route("**/api/conversations", async (route) => {
    if (route.request().method() === "POST") {
      if (rejectNextCreate) {
        rejectNextCreate = false;
        await route.fulfill({
          status: 503,
          json: { detail: "Test startup failure" },
        });
        return;
      }
      await createGate;
    }
    await route.fallback();
  });

  try {
    await page.goto(`/automations/setup?automationId=${automationId}`);
    await page
      .getByTestId("automation-setup-name")
      .fill("Unsaved docs reviewer");
    await setChatInput(page, "Why does this automation ignore docs?");
    await page.getByTestId("submit-button").click();

    await expect(page.getByRole("alert")).toContainText("Test startup failure");
    await page.getByRole("button", { name: "Edit", exact: true }).click();
    await expect(page.getByTestId("automation-setup-name")).toHaveValue(
      "Unsaved docs reviewer",
    );
    await expect(page.getByTestId("chat-input")).toContainText(
      "Why does this automation ignore docs?",
    );
    await page.getByTestId("submit-button").click();

    await expect(page).toHaveURL(/\/conversations\/launch-/, {
      timeout: 5_000,
    });
    await expect(page.getByTestId("conversation-launch-view")).toBeVisible();
    await expect(page.getByTestId("user-message")).toContainText(
      "Why does this automation ignore docs?",
    );
    await expect(
      page
        .getByTestId("conversation-panel")
        .locator('a[href*="/conversations/launch-"]'),
    ).toBeVisible();
    await testInfo.attach("pending-automation-chat", {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    releaseCreate();
    await expect(page).toHaveURL(/\/conversations\/(?!launch-)[^/?#]+/, {
      timeout: 45_000,
    });
    conversationId = new URL(page.url()).pathname.split("/").pop();
    await waitForNonUserMessageText(
      page,
      "AUTOMATION_HELP_CONTEXT_RECEIVED",
      45_000,
    );
    const modelRequests = JSON.stringify(
      (await getMockLLMRequests(request)).filter(
        (r) => Array.isArray(r.tools) && r.tools.length > 0,
      ),
    );
    expect(modelRequests).toContain(automationId);
    expect(modelRequests).toContain("Unsaved docs reviewer");
    expect(modelRequests).toContain("Why does this automation ignore docs?");
    await expect(page.getByTestId("user-message")).not.toContainText(
      "Automation form context",
    );
    // The assistant has the edit session as well as the textual context.
    await expect(page.getByTestId("automation-setup-name")).toHaveValue(
      "Unsaved docs reviewer",
    );
    await testInfo.attach("automation-chat-ready", {
      body: await page.screenshot(),
      contentType: "image/png",
    });
  } finally {
    releaseCreate();
    if (conversationId) await deleteConversation(request, conversationId);
    await request.delete(`${BACKEND_URL}/api/automation/v1/${automationId}`, {
      headers,
    });
    await resetMockLLM(request);
  }
});
