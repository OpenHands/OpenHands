/**
 * Mock-LLM E2E: image attachment is embedded as base64 and forwarded to LLM.
 *
 * Covers the fix from PR #1106: relative working-dir paths no longer resolve
 * to a read-only root, so file uploads (and image attachments processed via
 * the same pipeline) land in the correct writable location.
 *
 * Flow:
 *   1. Configure mock LLM profile via the settings API (skips UI setup).
 *   2. Register a scripted trajectory: one text reply with IMAGE_REPLY_TOKEN.
 *   3. Attach a minimal 1×1 PNG to the home-page chat input via the hidden
 *      file input (`data-testid="upload-image-input"`).
 *   4. Wait for the submit button to become enabled (image processed into
 *      the Zustand store → canSubmit flips to true).
 *   5. Type "What is in this image?" and submit.
 *   6. After the conversation is created and the agent replies:
 *      a. Verify IMAGE_REPLY_TOKEN appears in the chat UI.
 *      b. Verify the user's conversation event has image_urls set.
 *      c. Verify at least one LLM completion call to the mock server
 *         included an image_url content block (base64 data: URL).
 */

import { test, expect } from "@playwright/test";
import {
  IMAGE_REPLY_TOKEN,
  MINIMAL_PNG_BASE64,
  waitForAgentMessageContaining,
  waitForNonUserMessageText,
  getMockLLMRequests,
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

const USER_MESSAGE = "What is in this image?";
const TRAJECTORY_NAME = "image-query";

// ── Test suite ────────────────────────────────────────────────────────────────

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM image upload", () => {
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

  // ── Main test ──────────────────────────────────────────────────────────────

  test("attaching an image embeds it as base64 in the LLM completion call", async ({
    page,
    request,
  }) => {
    // ── 1. Configure mock LLM via the Settings UI ──
    //    Use a vision-capable model name so litellm does not strip image_url
    //    content blocks when constructing the completion request.  The base_url
    //    still points at the local mock server; the model name is purely a hint
    //    to litellm about what content types the model accepts.

    await ensureMockLLMProfile(page, { model: "openai/gpt-4o" });

    // ── 2. Register and activate the trajectory ──
    //    The mock LLM ignores the request body, so we don't need the agent to
    //    "understand" the image — we just want a reply that proves the LLM was
    //    called and the conversation completed successfully.
    //
    //    ⚠️  Padding note (mirrors the automation test's pattern):
    //    Public skills are bundled from @openhands/extensions at build time.
    //    The agent-server may make one internal LLM call for skill-analysis
    //    before the agent loop starts, consuming one trajectory slot.
    //    Turn 0 is a throwaway empty response that absorbs this internal call.
    //    Turn 1 is the agent's actual reply (IMAGE_REPLY_TOKEN).
    //    Turn 2 is a safety buffer in case a follow-up internal call is made.

    await resetMockLLM(request); // clears request history too
    await registerTrajectory(request, TRAJECTORY_NAME, [
      { text: "" },              // 0: padding — absorbs any internal skill-activation call
      { text: IMAGE_REPLY_TOKEN }, // 1: agent's actual reply
      { text: "" },              // 2: safety buffer for any follow-up internal call
    ]);
    await activateTrajectory(request, TRAJECTORY_NAME);

    // ── 3. Navigate to the home page ──

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    // ── 4. Attach the test image via the hidden file input ──
    //    Playwright's setInputFiles works on hidden inputs without needing a
    //    visible click target.  The onChange handler (handleFileInputChange →
    //    handleUpload → processImages → addImages) runs asynchronously in
    //    React, so we wait for the submit button to become enabled afterwards.

    await page.locator('[data-testid="upload-image-input"]').setInputFiles({
      name: "test-image.png",
      mimeType: "image/png",
      buffer: Buffer.from(MINIMAL_PNG_BASE64, "base64"),
    });

    // Wait for the image to be processed (FileReader async + Zustand store
    // update).  Once images.length > 0 the submit button stops being disabled.
    await expect(page.getByTestId("submit-button")).not.toBeDisabled({
      timeout: 10_000,
    });

    // ── 5. Type the user message and submit ──
    //    setChatInput sets innerText directly so the chat doesn't need visible
    //    typing; the InputEvent dispatch keeps canSubmit in sync.

    await setChatInput(page, USER_MESSAGE);

    // Confirm submit button is still enabled after text is added
    await expect(page.getByTestId("submit-button")).not.toBeDisabled({
      timeout: 5_000,
    });

    await page.getByTestId("submit-button").click();

    // ── 6a. Wait for the conversation page ──

    await waitForPath(page, /\/conversations\/.+/, 30_000);
    conversationId = getConversationIdFromURL(page);

    // ── 6b. Verify agent reply appears in the chat UI ──

    await test.step("agent reply token appears in chat UI", async () => {
      await waitForNonUserMessageText(page, IMAGE_REPLY_TOKEN, 30_000);
    });

    // ── 6c. Verify agent reply captured in conversation events API ──

    await test.step("agent reply captured in conversation events", async () => {
      await waitForAgentMessageContaining(
        request,
        conversationId!,
        IMAGE_REPLY_TOKEN,
        15_000,
      );
    });

    // ── 6d. Verify user message includes image_urls in conversation events ──
    //    The agent-server stores the user's message as a MessageEvent whose
    //    content array contains an "image" block with base64 data URLs.

    await test.step("user message event includes image_urls", async () => {
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
            const body = (await resp.json()) as { items?: unknown[] };
            const items = body.items ?? [];
            lastDiag = `${items.length} events`;

            return items.some((e: any) => {
              if (e.source !== "user") return false;
              // Check llm_message.content for image blocks (REST send path)
              if (Array.isArray(e.llm_message?.content)) {
                if (
                  e.llm_message.content.some(
                    (c: any) =>
                      c.type === "image" &&
                      Array.isArray(c.image_urls) &&
                      c.image_urls.length > 0,
                  )
                )
                  return true;
              }
              // Fall back: args.image_urls (WebSocket send path)
              const imageUrls = e.args?.image_urls;
              return Array.isArray(imageUrls) && imageUrls.length > 0;
            });
          },
          { timeout: 15_000 },
        )
        .toBe(true)
        .catch((err) => {
          throw new Error(
            `User message event should have image_urls after 15s.\n${lastDiag}`,
            { cause: err },
          );
        });
    });

    // ── 6e. Verify the LLM completion call included image content ──
    //    The mock server stores every /v1/chat/completions request body since
    //    the last reset.  At least one should contain an image_url content
    //    block with a base64 data: URL, confirming the frontend embedded the
    //    image rather than dropping it.

    await test.step("LLM completion call included image_url content", async () => {
      const llmRequests = await getMockLLMRequests(request);

      expect(
        llmRequests.length,
        "mock LLM should have received at least one completion request",
      ).toBeGreaterThan(0);

      // Helper: recursively walk any JSON value looking for a base64 image URL.
      function containsImageUrl(value: unknown): boolean {
        if (typeof value === "string") {
          return value.startsWith("data:image/");
        }
        if (Array.isArray(value)) {
          return value.some(containsImageUrl);
        }
        if (value !== null && typeof value === "object") {
          return Object.values(value as Record<string, unknown>).some(
            containsImageUrl,
          );
        }
        return false;
      }

      const anyRequestHadImage = llmRequests.some(containsImageUrl);
      expect(
        anyRequestHadImage,
        `At least one LLM completion call should include a base64 image data: URL.\n` +
          `Received ${llmRequests.length} request(s).\n` +
          llmRequests
            .map(
              (req, i) =>
                `Request ${i} messages:\n` +
                JSON.stringify((req as any)?.messages ?? [], null, 2).slice(
                  0,
                  800,
                ),
            )
            .join("\n---\n"),
      ).toBe(true);
    });

    // ── 6f. Verify no error banners appeared ──

    await test.step("no error banners", async () => {
      await expect(page.getByTestId("error-message-banner")).not.toBeVisible({
        timeout: 2_000,
      });
    });
  });
});
