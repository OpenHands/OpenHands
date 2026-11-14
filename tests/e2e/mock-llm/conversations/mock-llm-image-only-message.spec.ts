/**
 * Mock-LLM E2E: an image-only conversation message (empty caption) is
 * delivered over the conversation WebSocket and acknowledged.
 *
 * Covers issue #17398: `useSendMessage` used to skip sending whenever the
 * caption was empty, so an image-only message never reached the agent-server
 * and its pending bubble stayed at "Sending…" until the pending-message
 * watchdog failed it ("Send timed out"). The fix sends when text OR images
 * are present and omits the empty text block for image-only messages.
 *
 * Flow:
 *   1. Configure the mock LLM profile via the settings API.
 *   2. Register a scripted trajectory: one reply for the captioned first
 *      message, one for the image-only follow-up.
 *   3. Start a conversation from the home page with a captioned text message
 *      (control: text path unchanged).
 *   4. In the conversation composer, attach a PNG and submit with the text
 *      box left empty (the WebSocket send path under test).
 *   5. Verify the agent's second reply arrives (delivery + acknowledgement).
 *   6. Verify the stored user message event has the image block and no empty
 *      text block, and that no pending-message timeout error appeared.
 */

import { test, expect } from "@playwright/test";
import {
  MINIMAL_PNG_BASE64,
  waitForAgentMessageContaining,
  waitForNonUserMessageText,
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  deleteConversation,
  resetMockLLM,
  registerTrajectory,
  activateTrajectory,
  ensureMockLLMProfile,
  setChatInput,
} from "../utils/mock-llm-helpers";

// ── Constants ────────────────────────────────────────────────────────────────

const FIRST_MESSAGE = "Please confirm you can see my next attachment.";
const FIRST_REPLY_TOKEN = "FIRST_MESSAGE_ACK_TOKEN";
const IMAGE_ONLY_REPLY_TOKEN = "IMAGE_ONLY_ACK_TOKEN";
const TRAJECTORY_NAME = "image-only-message";

interface UserEvent {
  source?: string;
  llm_message?: { content?: unknown };
}

// ── Test suite ────────────────────────────────────────────────────────────────

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM image-only conversation message", () => {
  let conversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    if (conversationId) {
      try {
        await deleteConversation(request, conversationId);
      } catch {
        // best-effort cleanup
      }
      conversationId = null;
    }
    await resetMockLLM(request);
  });

  test("an image-only message is sent, acknowledged, and stored without an empty text block", async ({
    page,
    request,
  }) => {
    // ── 1. Configure the mock LLM profile (vision-capable model hint) ──

    await ensureMockLLMProfile(page, { model: "openai/gpt-4o" });

    // ── 2. Register and activate the trajectory ──
    //    Turn 0 is padding for the agent-server's internal LLM call (title
    //    generation / skill activation). Turns 1-2 answer the two user
    //    messages; turn 3 is a safety buffer.

    await resetMockLLM(request); // clears request history too
    await registerTrajectory(request, TRAJECTORY_NAME, [
      { text: "" }, // 0: padding — absorbs the internal call
      { text: FIRST_REPLY_TOKEN }, // 1: reply to the captioned message
      { text: IMAGE_ONLY_REPLY_TOKEN }, // 2: reply to the image-only message
      { text: "" }, // 3: safety buffer
    ]);
    await activateTrajectory(request, TRAJECTORY_NAME);

    // ── 3. Start a conversation with a captioned text message ──

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    await setChatInput(page, FIRST_MESSAGE);
    await page.getByTestId("submit-button").click();

    await waitForPath(page, /\/conversations\/.+/, 30_000);
    conversationId = getConversationIdFromURL(page);

    // The agent answers the captioned message, proving the conversation and
    // the text send path work before the image-only case is exercised.
    await waitForNonUserMessageText(page, FIRST_REPLY_TOKEN, 30_000);

    // ── 4. Attach an image in the conversation composer, leave text empty ──

    await page.locator('[data-testid="upload-image-input"]').setInputFiles({
      name: "image-only.png",
      mimeType: "image/png",
      buffer: Buffer.from(MINIMAL_PNG_BASE64, "base64"),
    });

    // The composer enables submit for attachments alone (no caption typed).
    await expect(page.getByTestId("submit-button")).not.toBeDisabled({
      timeout: 10_000,
    });
    await page.getByTestId("submit-button").click();

    // ── 5. The agent's reply acknowledges the image-only message ──
    //    Before the fix this never arrived: the hook skipped the empty
    //    caption, the bubble stayed at "Sending…" and the watchdog eventually
    //    failed it with "Send timed out".

    await test.step("agent acknowledges the image-only message", async () => {
      await waitForNonUserMessageText(page, IMAGE_ONLY_REPLY_TOKEN, 30_000);
    });

    await test.step("no pending-send timeout error is shown", async () => {
      await expect(page.getByText("Send timed out")).not.toBeVisible({
        timeout: 2_000,
      });
    });

    // ── 6. The stored user event carries the image block, no empty text ──

    await test.step("user event has the image block and no empty text block", async () => {
      let lastDiag = "no polls yet";
      await expect
        .poll(
          async () => {
            const resp = await request.get(
              `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId!)}/events/search`,
              {
                headers: { "X-Session-API-Key": SESSION_API_KEY },
                params: { limit: "50", sort_order: "TIMESTAMP_DESC" },
              },
            );
            if (!resp.ok()) {
              lastDiag = `events API: ${resp.status()}`;
              return false;
            }
            const body = (await resp.json()) as { items?: UserEvent[] };
            const items = body.items ?? [];
            lastDiag = `${items.length} events`;

            const imageOnlyEvent = items.find((e) => {
              if (e.source !== "user") return false;
              const content = e.llm_message?.content;
              if (!Array.isArray(content)) return false;
              const hasImage = content.some(
                (c: any) => c.type === "image" && c.image_urls?.length > 0,
              );
              const hasEmptyText = content.some(
                (c: any) => c.type === "text" && c.text === "",
              );
              return hasImage && !hasEmptyText;
            });
            if (imageOnlyEvent) {
              const content = imageOnlyEvent.llm_message?.content as any[];
              lastDiag = JSON.stringify(content).slice(0, 400);
            }
            return Boolean(imageOnlyEvent);
          },
          { timeout: 15_000 },
        )
        .toBe(true)
        .catch((err) => {
          throw new Error(
            `Image-only user event should carry an image block and no empty text block after 15s.\n${lastDiag}`,
            { cause: err },
          );
        });
    });

    // ── 7. The agent-server captured the acknowledged reply ──

    await test.step("acknowledged reply captured in conversation events", async () => {
      await waitForAgentMessageContaining(
        request,
        conversationId!,
        IMAGE_ONLY_REPLY_TOKEN,
        15_000,
      );
    });

    // ── 8. No error banners ──

    await test.step("no error banners", async () => {
      await expect(page.getByTestId("error-message-banner")).not.toBeVisible({
        timeout: 2_000,
      });
    });
  });
});
