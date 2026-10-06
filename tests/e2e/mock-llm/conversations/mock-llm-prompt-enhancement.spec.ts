/**
 * Mock-LLM E2E: "Enhance prompt" in the home composer (issue #17496).
 *
 * Exercises the real Agent Server prompt-enhancement endpoint
 * (`/api/prompt-enhancement/*`, capability `prompt_enhancement_v1`) with a
 * scripted model reply:
 *   1. Enhancing a draft shows an editable suggestion and leaves the draft,
 *      the route, and the conversation list unchanged.
 *   2. The model call is a single tool-free completion of the draft.
 *   3. "Use enhanced prompt" replaces the draft with the edited suggestion,
 *      line breaks and code blocks unchanged, without sending it.
 *   4. "Keep original" closes a second suggestion and keeps the draft.
 */

import { test, expect, type Page } from "@playwright/test";
import {
  activateTrajectory,
  dismissAnalyticsModal,
  ensureMockLLMProfile,
  getMockLLMRequests,
  registerTrajectory,
  resetMockLLM,
  routeSessionApiKey,
  seedLocalStorage,
  setChatInput,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const DRAFT = "fix login bug in src/auth.ts, dont break the 3 existing tests";
// A code block with a blank line: accepting must keep the text exact.
const ENHANCED = [
  "Fix the login bug in `src/auth.ts`.",
  "",
  "Keep this check:",
  "```ts",
  "if (expired) {",
  "",
  "  return 401;",
  "}",
  "```",
  "Do not break the 3 existing tests.",
].join("\n");
const EDIT = " Add a regression test.";
const SECOND_ENHANCED = "A second suggestion that the user rejects.";

const chatInputText = (page: Page) =>
  page
    .getByTestId("chat-input")
    .evaluate((el) => (el as HTMLElement).innerText);

async function enhanceDraft(page: Page) {
  const button = page.getByTestId("chat-enhance-prompt-button");
  // The availability check runs once per backend and profile.
  await expect(button).toHaveAttribute("aria-disabled", "false", {
    timeout: 30_000,
  });
  await button.click();
}

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM prompt enhancement", () => {
  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    await resetMockLLM(request);
  });

  test("reviews, edits, and accepts or rejects an enhanced draft", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page, { profileName: "mock-llm" });
    await resetMockLLM(request);
    await registerTrajectory(request, "prompt-enhancement", [
      { text: ENHANCED },
      { text: SECOND_ENHANCED },
    ]);
    await activateTrajectory(request, "prompt-enhancement");

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");
    await setChatInput(page, DRAFT);

    await test.step("show an editable suggestion without side effects", async () => {
      await enhanceDraft(page);
      const suggestion = page.getByTestId("prompt-enhancement-text");
      await expect(suggestion).toHaveValue(ENHANCED, { timeout: 30_000 });
      expect(await chatInputText(page)).toBe(DRAFT);
      expect(new URL(page.url()).pathname).toBe("/");

      const completions = await getMockLLMRequests(request);
      expect(completions).toHaveLength(1);
      const [completion] = completions;
      expect(completion.tools ?? []).toEqual([]);
      expect(JSON.stringify(completion.messages)).toContain(DRAFT);
    });

    await test.step("accept the edited suggestion without sending it", async () => {
      // The suggestion has focus with the caret at its end.
      await page.keyboard.type(EDIT);
      await page.getByTestId("prompt-enhancement-use").click();

      await expect(page.getByTestId("prompt-enhancement-panel")).toHaveCount(0);
      await expect.poll(() => chatInputText(page)).toBe(`${ENHANCED}${EDIT}`);
      expect(new URL(page.url()).pathname).toBe("/");
    });

    await test.step("keep the original draft", async () => {
      await enhanceDraft(page);
      await expect(page.getByTestId("prompt-enhancement-text")).toHaveValue(
        SECOND_ENHANCED,
        { timeout: 30_000 },
      );
      await page.getByTestId("prompt-enhancement-keep-original").click();

      await expect(page.getByTestId("prompt-enhancement-panel")).toHaveCount(0);
      expect(await chatInputText(page)).toBe(`${ENHANCED}${EDIT}`);
    });
  });
});
