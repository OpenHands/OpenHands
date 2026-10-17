/**
 * Standalone reproduction for issue #18445.
 *
 * Drives the REAL built Canvas bundle + REAL agent-server session socket,
 * mocking only the cloud API responses that create the bug condition:
 *   - GET /api/v1/app-conversations        -> the completed cloud conversation
 *   - GET /api/v1/conversation/{id}/events/search -> 500 (the failing tail load)
 *   - GET /api/conversations/{id}/events/count    -> passes through (real 200)
 *   - ws://.../sockets/session/{id}               -> real agent-server replay
 *
 * Usage: node repro-18445.mjs <baseUrl> <conversationId> <outPrefix>
 */
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:18300";
const conversationId = process.argv[3];
const outPrefix = process.argv[4] ?? "/tmp/repro";
const SESSION_KEY = "e2e-secret-key";

const cloudConversation = {
  id: conversationId,
  conversation_id: conversationId,
  created_by_user_id: null,
  sandbox_id: "sandbox-demo",
  title: "Completed demo conversation",
  llm_model: "openai/mock-test-model",
  metrics: null,
  created_at: "2026-10-14T09:00:00Z",
  updated_at: "2026-10-14T09:30:00Z",
  execution_status: "idle",
  sandbox_status: "RUNNING",
  conversation_url: `${baseUrl}/api/conversations/${conversationId}`,
  session_api_key: SESSION_KEY,
  workspace: null,
  sub_conversation_ids: [],
};

const browser = await chromium.launch();
const context = await browser.newContext({
  recordVideo: { dir: `${outPrefix}-video` },
});
const page = await context.newPage();

const wsUrls = [];
page.on("websocket", (ws) => {
  if (ws.url().includes("/sockets/session/")) wsUrls.push(ws.url());
});

await page.addInitScript(
  ({ id }) => {
    window.localStorage.setItem("analytics-consent", "false");
    window.localStorage.setItem("openhands-telemetry-consent", "denied");
    window.localStorage.setItem("openhands-telemetry-first-use", "true");
    window.localStorage.setItem("openhands-onboarded", "1");
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([
        {
          id: "cloud-18445",
          name: "OpenHands Cloud (repro)",
          host: window.location.origin,
          apiKey: "cloud-repro-api-key",
          kind: "cloud",
        },
      ]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "cloud-18445", orgId: "repro-org" }),
    );
  },
  { id: conversationId },
);

// Catch-all first (LIFO: specific routes registered later win).
await page.route("**/api/v1/**", async (route) => {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({ items: [], profiles: [], organizations: [] }),
  });
});

await page.route("**/api/v1/app-conversations**", async (route) => {
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify([cloudConversation]),
  });
});

// The failing initial history tail load — the bug trigger.
await page.route("**/api/v1/conversation/*/events/search**", async (route) => {
  await route.fulfill({
    status: 500,
    contentType: "application/json",
    body: JSON.stringify({ detail: "Internal Server Error" }),
  });
});

// Pass runtime count + sockets through to the real agent-server with the key.
await page.route("**/api/conversations/**", async (route) => {
  await route.continue({
    headers: {
      ...route.request().headers(),
      "X-Session-API-Key": SESSION_KEY,
    },
  });
});

await page.goto(`${baseUrl}/conversations/${conversationId}`, {
  waitUntil: "domcontentloaded",
});

// Give the socket time to connect and replay.
await page.waitForTimeout(8000);

const userMessages = await page
  .locator('[data-testid="user-message"]')
  .allInnerTexts();
const userMessagesText = await page
  .locator('[data-testid="user-message"]')
  .evaluateAll((els) => els.map((el) => el.textContent ?? ""));
const agentMessages = await page
  .locator('[data-testid="agent-message"]')
  .allInnerTexts();
const chatText = await page.locator('[data-testid="chat-interface"]').innerText();
const chatTextContent = await page
  .locator('[data-testid="chat-interface"]')
  .evaluate((el) => el.textContent ?? "");
const seededVisible =
  chatText.includes("Demo prior user turn 1") ||
  chatTextContent.includes("Demo prior user turn 1");

const chatHtml = await page
  .locator('[data-testid="chat-interface"]')
  .evaluate((el) => el.outerHTML);

await page.screenshot({ path: `${outPrefix}.png`, fullPage: false });

const result = {
  seededVisible,
  userMessageCount: userMessages.length,
  agentMessageCount: agentMessages.length,
  userMessages,
  userMessagesText,
  chatHtmlPreview: chatHtml.slice(0, 1500),
  wsUrls,
};
writeFileSync(`${outPrefix}.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));

await context.close();
await browser.close();
