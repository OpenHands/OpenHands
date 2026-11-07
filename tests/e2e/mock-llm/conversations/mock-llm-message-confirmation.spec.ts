import { expect, test, type Page } from "@playwright/test";
import {
  activateTrajectory,
  BACKEND_URL,
  deleteConversation,
  ensureMockLLMProfile,
  getConversationIdFromURL,
  registerTrajectory,
  resetMockLLM,
  seedLocalStorage,
  SESSION_API_KEY,
  setChatInput,
} from "../utils/mock-llm-helpers";

const PROMPT = "Continue with the validation.";
const REPLY = "Message confirmation completed.";

async function sendMessage(page: Page) {
  await setChatInput(page, PROMPT);
  await page.getByTestId("submit-button").click();
}

async function userEventCount(page: Page, conversationId: string) {
  const response = await page.request.get(
    `${BACKEND_URL}/api/conversations/${conversationId}/events/search?limit=100`,
    { headers: { "X-Session-API-Key": SESSION_API_KEY } },
  );
  expect(response.ok()).toBe(true);
  const body = await response.json();
  return body.items.filter(
    (event: { source: string }) => event.source === "user",
  ).length;
}

test.beforeEach(async ({ page, request }) => {
  await seedLocalStorage(page);
  await ensureMockLLMProfile(page, { profileName: "message-confirmation" });
  await registerTrajectory(
    request,
    "message-confirmation",
    Array.from({ length: 8 }, () => ({
      tool_call: { name: "finish", arguments: { message: REPLY } },
    })),
  );
  await activateTrajectory(request, "message-confirmation");
  await page.goto("/");
  await sendMessage(page);
  await page.waitForURL(/\/conversations\/[^/?]+/);
  await expect(page.getByText(REPLY, { exact: true }).last()).toBeVisible();
  await expect(page.getByTestId("chat-message-sending")).toHaveCount(0);
});

test.afterEach(async ({ page, request }) => {
  const match = page.url().match(/\/conversations\/([^/?#]+)/);
  if (match) await deleteConversation(request, decodeURIComponent(match[1]));
  await resetMockLLM(request);
});

test("confirms a real server echo when the browser clock is ahead", async ({
  page,
}) => {
  const conversationId = getConversationIdFromURL(page);
  await page.clock.setFixedTime(new Date(Date.now() + 60_000));

  await sendMessage(page);

  await expect.poll(() => userEventCount(page, conversationId)).toBe(2);
  await expect(page.getByTestId("chat-message-sending")).toHaveCount(0);
  await expect(page.getByTestId("chat-message-retry")).toHaveCount(0);
});

test("preserves the failed attempt through history reload, identical send and retry", async ({
  page,
}) => {
  const conversationId = getConversationIdFromURL(page);
  await page.evaluate(() => {
    const send = WebSocket.prototype.send;
    WebSocket.prototype.send = function (data) {
      if (typeof data === "string" && JSON.parse(data)?.role === "user") {
        WebSocket.prototype.send = send;
        throw new DOMException("Controlled send failure", "NetworkError");
      }
      return send.call(this, data);
    };
  });
  await sendMessage(page);
  await expect(page.getByTestId("chat-message-retry")).toHaveCount(1);

  await page.getByRole("link", { name: "New Chat", exact: true }).click();
  await expect(page.getByTestId("home-chat-launcher")).toBeVisible();
  const history = page.waitForResponse(
    (response) =>
      response
        .url()
        .includes(`/conversations/${conversationId}/events/search`) &&
      response.ok(),
  );
  await page
    .locator(`a[href*="/conversations/${conversationId}"]`)
    .first()
    .click();
  await history;
  await expect(page.getByTestId("chat-message-retry")).toHaveCount(1);

  await sendMessage(page);
  await expect.poll(() => userEventCount(page, conversationId)).toBe(2);
  await expect(page.getByTestId("chat-message-sending")).toHaveCount(0);
  await expect(page.getByTestId("chat-message-retry")).toHaveCount(1);

  await page.getByTestId("chat-message-retry").click();
  await expect.poll(() => userEventCount(page, conversationId)).toBe(3);
  await expect(page.getByTestId("chat-message-retry")).toHaveCount(0);
  await expect(page.getByTestId("chat-message-sending")).toHaveCount(0);
});
