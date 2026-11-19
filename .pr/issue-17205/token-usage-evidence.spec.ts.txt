/**
 * Evidence capture for issue #17205 (not committed).
 *
 * Runs a scripted multi-tool conversation (more than 50 events) through the
 * real agent server, reloads the page so the chat holds only the newest page
 * of events, and captures the Usage tab's "Token usage by activity" card.
 */
import { test, expect } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  activateTrajectory,
  dismissAnalyticsModal,
  ensureMockLLMProfile,
  getConversationIdFromURL,
  registerTrajectory,
  routeSessionApiKey,
  seedLocalStorage,
  setChatInput,
  waitForAgentMessageContaining,
  waitForPath,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const OUT_DIR = process.env.EVIDENCE_DIR ?? ".pr-evidence";
const REPLY = "The ice cream page is ready in index.html.";

const terminal = (command: string) => ({
  tool_call: { name: "terminal", arguments: { command } },
});

test("token usage by activity in the Usage tab", async ({ page, request }) => {
  test.setTimeout(420_000);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await seedLocalStorage(page);
  await ensureMockLLMProfile(page, { profileName: "mock-llm" });

  const flavors = ["vanilla", "chocolate", "mango", "pistachio", "lemon"];
  await registerTrajectory(request, "token-usage-evidence", [
    // Consumed by the conversation title request.
    { text: "Ice cream web page" },
    {
      tool_call: {
        name: "task_tracker",
        arguments: {
          command: "plan",
          task_list: [
            { title: "Create the page", notes: "", status: "in_progress" },
            { title: "Add the flavors", notes: "", status: "todo" },
            { title: "Check the page", notes: "", status: "todo" },
          ],
        },
      },
    },
    terminal("printf '<h1>Ice cream</h1>\\n' > index.html"),
    ...flavors.flatMap((flavor) => [
      terminal(`printf '<p>${flavor}</p>\\n' >> index.html`),
      terminal(`grep -c '${flavor}' index.html`),
      {
        tool_call: {
          name: "think",
          arguments: { thought: `The ${flavor} entry is in the page.` },
        },
      },
    ]),
    ...flavors.map((flavor) => terminal(`grep -n '${flavor}' index.html`)),
    terminal("cat index.html"),
    terminal("wc -l index.html"),
    { text: REPLY },
  ]);
  await activateTrajectory(request, "token-usage-evidence");

  await routeSessionApiKey(page);
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "home-chat-launcher");
  await setChatInput(page, "Build a small ice cream web page.");
  await page.getByTestId("submit-button").click();
  await waitForPath(page, /\/conversations\/.+/, 60_000);
  const conversationId = getConversationIdFromURL(page);
  console.log("CONVERSATION_ID", conversationId);
  await waitForAgentMessageContaining(request, conversationId, REPLY, 240_000);

  // Reload: the chat now holds only its newest page of events, so the card
  // must load the older history to attribute the first calls.
  const historyRequests: string[] = [];
  page.on("request", (req) => {
    const url = new URL(req.url());
    if (url.pathname.endsWith("/events/search")) {
      historyRequests.push(url.search);
    }
  });
  await page.reload({ waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);

  await page.getByTestId("right-panel-toggle").click();
  await page.getByTestId("conversation-tab-usage").click();
  const card = page.getByTestId("token-usage-breakdown");
  await expect(card).toBeVisible({ timeout: 60_000 });
  await expect(
    card.locator('[data-activity-key="tools:task_tracker"]'),
  ).toBeVisible({ timeout: 60_000 });
  await expect(card.locator('[data-activity-key="unattributed"]')).toHaveCount(
    0,
  );

  const headers = { "X-Session-API-Key": SESSION_API_KEY };
  const info = await (
    await request.get(`${BACKEND_URL}/api/conversations/${conversationId}`, {
      headers,
    })
  ).json();
  const eventCount = await (
    await request.get(
      `${BACKEND_URL}/api/conversations/${conversationId}/events/count`,
      { headers },
    )
  ).json();
  const usages = Object.values(
    info.stats.usage_to_metrics as Record<
      string,
      { token_usages: Array<Record<string, number | string>> }
    >,
  ).flatMap((metrics) => metrics.token_usages);
  console.log("EVENT_COUNT", eventCount);
  console.log("CALLS", usages.length);
  console.log(
    "SERVER_TOTAL",
    usages.reduce(
      (sum, u) =>
        sum + Number(u.prompt_tokens ?? 0) + Number(u.completion_tokens ?? 0),
      0,
    ),
  );
  console.log("HISTORY_REQUESTS", JSON.stringify(historyRequests));
  console.log(
    "CARD_ROWS",
    JSON.stringify(
      await card
        .getByTestId("token-usage-activity")
        .evaluateAll((rows) => rows.map((row) => row.textContent)),
    ),
  );

  await page.screenshot({ path: `${OUT_DIR}/17205-conversation-usage-tab.png` });
  await card.screenshot({ path: `${OUT_DIR}/17205-token-usage-card.png` });

  // Same card in a light theme: the bars use theme tokens, not fixed colors.
  await page.evaluate(() =>
    window.localStorage.setItem("openhands-color-theme", "light-plus"),
  );
  await page.reload({ waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  const lightCard = page.getByTestId("token-usage-breakdown");
  // The drawer is session-only and starts closed after a reload; the
  // selected tab (Usage) is restored, so only the drawer is opened.
  await page.getByTestId("right-panel-toggle").click();
  await expect(
    lightCard.locator('[data-activity-key="tools:task_tracker"]'),
  ).toBeVisible({ timeout: 60_000 });
  await expect
    .poll(async () => (await lightCard.boundingBox())?.width ?? 0)
    .toBeGreaterThan(400);
  await page.waitForTimeout(1_000);
  await page.screenshot({
    path: `${OUT_DIR}/17205-conversation-usage-tab-light.png`,
  });
});
