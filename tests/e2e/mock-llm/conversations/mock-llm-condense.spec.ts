import { expect, test } from "@playwright/test";
import {
  activateTrajectory,
  BACKEND_URL,
  deleteConversation,
  dismissAnalyticsModal,
  ensureMockLLMProfile,
  getConversationIdFromURL,
  registerTrajectory,
  resetMockLLM,
  routeSessionApiKey,
  seedLocalStorage,
  SESSION_API_KEY,
  setChatInput,
  waitForNonUserMessageText,
  waitForPath,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const TRAJECTORY_NAME = "manual-condensation";
const PROFILE_NAME = "mock-llm-condense";
const CONTEXT_MESSAGES = [
  "Remember that the project uses React.",
  "Remember that tests use Vitest.",
  "Remember that the UI supports slash commands.",
  "Remember that condensation should preserve the key facts.",
];
const AGENT_REPLIES = [
  "CONTEXT_TURN_1_OK",
  "CONTEXT_TURN_2_OK",
  "CONTEXT_TURN_3_OK",
  "CONTEXT_TURN_4_OK",
];

test("/condense summarizes a conversation with enough history", async ({
  page,
  request,
}) => {
  let conversationId: string | null = null;

  await seedLocalStorage(page);
  await resetMockLLM(request);
  await ensureMockLLMProfile(page, { profileName: PROFILE_NAME });
  await registerTrajectory(request, TRAJECTORY_NAME, [
    // The first completion generates the conversation title.
    { text: "Condensation demo" },
    ...AGENT_REPLIES.map((text) => ({ text })),
    { text: "The project uses React and Vitest with slash-command support." },
    { text: "CONDENSATION_STEP_OK" },
  ]);
  await activateTrajectory(request, TRAJECTORY_NAME);

  await routeSessionApiKey(page);
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "home-chat-launcher");

  try {
    for (let index = 0; index < CONTEXT_MESSAGES.length; index += 1) {
      await setChatInput(page, CONTEXT_MESSAGES[index]);
      await page.getByTestId("submit-button").click();

      if (index === 0) {
        await waitForPath(page, /\/conversations\/.+/, 30_000);
        conversationId = getConversationIdFromURL(page);
      }

      await waitForNonUserMessageText(page, AGENT_REPLIES[index], 30_000);
    }

    const chatInput = page.getByTestId("chat-input");
    await chatInput.click();
    await chatInput.pressSequentially("/condense", { delay: 100 });
    await expect(chatInput).toContainText("/condense");
    await page.getByTestId("submit-button").click();

    await expect(
      page.getByText("Conversation history condensed.", { exact: true }),
    ).toBeVisible({ timeout: 30_000 });

    await expect
      .poll(
        async () => {
          const response = await request.get(
            `${BACKEND_URL}/api/conversations/${conversationId}/events/search`,
            {
              headers: { "X-Session-API-Key": SESSION_API_KEY },
              params: { limit: "100", sort_order: "TIMESTAMP_DESC" },
            },
          );
          if (!response.ok()) return false;
          const body = (await response.json()) as { items?: unknown[] };
          return (body.items ?? []).some(
            (event: any) =>
              event.kind === "Condensation" &&
              Array.isArray(event.forgotten_event_ids) &&
              event.forgotten_event_ids.length > 0,
          );
        },
        { timeout: 30_000 },
      )
      .toBe(true);
  } finally {
    if (conversationId) await deleteConversation(request, conversationId);
    await resetMockLLM(request);
  }
});
