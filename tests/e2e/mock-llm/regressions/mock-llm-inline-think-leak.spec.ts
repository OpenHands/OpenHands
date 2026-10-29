/**
 * Mock-LLM E2E regression: inline reasoning in an action thought must not leak
 * into the chat bubble.
 *
 * Some models stream their reasoning inline in the assistant content instead of
 * via `reasoning_content`. When that turn ends in a tool call the agent-server
 * stores the content as the `ActionEvent.thought`, and Canvas used to render it
 * verbatim in the agent bubble — so the same words read as duplicated once the
 * final reply arrived. The reasoning now renders in the collapsible thinking
 * section instead.
 *
 * Reproduces github.com/OpenHands/OpenHands#18074.
 */

import { test, expect, Page } from "@playwright/test";
import {
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

const USER_MESSAGE = "Run a quick terminal command and then reply.";
const TRAJECTORY_NAME = "inline-think-leak";
const REPLY_TOKEN = "MOCK_LLM_INLINE_THINK_REPLY_OK";
const REASONING_TEXT = "Let me check the working directory before running it.";
const THOUGHT_WITH_TAGS = `<think>${REASONING_TEXT}</think>\nRunning the command now.`;

interface ActionThoughtEvent {
  kind: string;
  thought?: Array<{ text?: string }>;
}

interface LeakProbe {
  bubbleHasReasoning: boolean;
  bubbleHasRawTag: boolean;
  thinkingSections: number;
}

/**
 * Inspect the rendered conversation for a leak of the reasoning text into a
 * visible agent bubble, and count the collapsible thinking sections.
 */
async function probeLeak(page: Page, reasoning: string): Promise<LeakProbe> {
  return page.evaluate((expected) => {
    const agentBubbles = Array.from(
      document.querySelectorAll('[data-testid="agent-message"]'),
    );
    const visible = agentBubbles.filter(
      (el) =>
        (el as HTMLElement).offsetWidth > 0 ||
        (el as HTMLElement).offsetHeight > 0,
    );
    const body = visible.map((el) => el.textContent ?? "").join("\n");
    return {
      bubbleHasReasoning: body.includes(expected),
      bubbleHasRawTag: body.includes("think>"),
      thinkingSections: document.querySelectorAll(
        '[data-testid="collapsible-thinking"]',
      ).length,
    };
  }, reasoning);
}

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM inline-think leak", () => {
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

  test("an inline reasoning block in a tool-call thought stays out of the bubble", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    // Turn 0 absorbs any internal skill-analysis call the agent-server makes
    // before the agent loop; turn 1 is the tool call carrying the thought.
    await resetMockLLM(request);
    await registerTrajectory(request, TRAJECTORY_NAME, [
      { text: "" },
      {
        tool_call: {
          name: "terminal",
          arguments: { command: `printf 'inline-think-ok\\n'` },
          content: THOUGHT_WITH_TAGS,
        },
      },
      { text: REPLY_TOKEN },
      { text: "" },
    ]);
    await activateTrajectory(request, TRAJECTORY_NAME);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    await setChatInput(page, USER_MESSAGE);
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    conversationId = getConversationIdFromURL(page);

    // The agent-server persists the tool-call turn's content as the thought.
    await expect
      .poll(
        async () => {
          const resp = await request.get(
            `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId!)}/events/search`,
            {
              headers: { "X-Session-API-Key": SESSION_API_KEY },
              params: { limit: "100", sort_order: "TIMESTAMP_DESC" },
            },
          );
          if (!resp.ok()) return false;
          const body = (await resp.json()) as {
            items?: ActionThoughtEvent[];
          };
          const items = body.items ?? [];
          return items.some(
            (e) =>
              e.kind === "ActionEvent" &&
              Array.isArray(e.thought) &&
              e.thought.some((t) => (t.text ?? "").includes(REASONING_TEXT)),
          );
        },
        { timeout: 30_000 },
      )
      .toBe(true);

    const bubbleWithThought = page
      .locator('[data-testid="agent-message"]', {
        hasText: "Running the command now.",
      })
      .first();

    await test.step("live turn: reasoning is collapsed, bubble is clean", async () => {
      await expect(bubbleWithThought).toBeVisible({ timeout: 30_000 });
      await expect(
        page.getByTestId("collapsible-thinking").first(),
      ).toBeVisible({ timeout: 30_000 });

      const probe = await probeLeak(page, REASONING_TEXT);
      expect(probe.thinkingSections).toBe(1);
      expect(probe.bubbleHasReasoning).toBe(false);
      expect(probe.bubbleHasRawTag).toBe(false);
    });

    await test.step("after reload: the settled turn renders identically", async () => {
      await page.reload({ waitUntil: "domcontentloaded" });
      await expect(bubbleWithThought).toBeVisible({ timeout: 30_000 });

      const probe = await probeLeak(page, REASONING_TEXT);
      expect(probe.thinkingSections).toBe(1);
      expect(probe.bubbleHasReasoning).toBe(false);
      expect(probe.bubbleHasRawTag).toBe(false);
    });
  });
});
