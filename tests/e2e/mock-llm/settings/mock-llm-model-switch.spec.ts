/**
 * Mock-LLM E2E test: /model slash command — mid-conversation LLM profile switching.
 *
 * Exercises the full /model flow end-to-end against the real agent-server:
 *
 *   1. Setup: configure the active LLM settings via ensureMockLLMProfile
 *      (the proven pattern), create a named profile B as the switch target
 *      via the profiles API, and register a trajectory with text replies.
 *
 *   2. Conversation + switch: start a conversation from the home page,
 *      wait for the agent to reply, then type `/model <profile-B>` in the
 *      chat input. Verify the "Switched to profile" confirmation renders in
 *      the chat UI. Verify the switch_llm POST was made to the agent-server
 *      (the frontend fetches the full encrypted profile config and sends it
 *      via /switch_llm rather than calling /switch_profile by name).
 *
 *   3. Post-switch verification: send another message after the switch
 *      and verify the agent responds, proving the conversation continues
 *      working under the new profile.
 */

import { expect } from "@playwright/test";
import { test } from "../utils/atomic-journey";
import {
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  registerTrajectory,
  activateTrajectory,
  ensureMockLLMProfile,
  createProfileViaUI,
  setChatInput,
} from "../utils/mock-llm-helpers";

/** Profile B is the switch target — created via the Settings UI. */
const MODEL_B = "openai/mock-model-beta";

const INITIAL_REPLY_TOKEN = "MODEL_SWITCH_INITIAL_REPLY_OK";
const POST_SWITCH_REPLY_TOKEN = "MODEL_SWITCH_POST_SWITCH_REPLY_OK";

test("create profiles and switch model during a conversation", async ({
  page,
  request,
  journey,
}) => {
  test.setTimeout(180_000);
  const PROFILE_B_NAME = journey.newProfileName("switch-target");
  await test.step("configure LLM, create switch-target profile, register trajectory", async () => {
    // Use the Settings UI to create + activate a mock LLM profile — the same
    // flow used by mock-llm-conversation.spec.ts.
    await ensureMockLLMProfile(page, { profileName: journey.profileName });

    // Create profile B as the switch target through the Settings UI — it has
    // a different model name but the same mock LLM base_url so post-switch
    // inference still works. ensureMockLLMProfile leaves this page ready.
    await createProfileViaUI(page, {
      profileName: PROFILE_B_NAME,
      model: MODEL_B,
    });

    // Verify profile B appears in the list
    const profileRows = page.getByTestId("profile-row");
    const profileTexts = await profileRows.allTextContents();
    expect(
      profileTexts.some((text) => text.includes(PROFILE_B_NAME)),
      `Profile "${PROFILE_B_NAME}" should appear in the list`,
    ).toBe(true);

    // Title requests are handled separately by the mock server. These
    // responses belong only to the initial and post-switch agent turns.
    await registerTrajectory(request, "model-switch", [
      { text: INITIAL_REPLY_TOKEN },
      { text: POST_SWITCH_REPLY_TOKEN },
    ]);
    await activateTrajectory(request, "model-switch");
  });

  await test.step("start conversation, switch profile via /model, verify switch", async () => {
    // Track whether the switch_llm POST was intercepted.
    // The frontend calls POST /api/conversations/{id}/switch_llm with the full
    // encrypted profile config (model + api_key + base_url) rather than calling
    // /switch_profile by name — this avoids an extra agent-server secrets fetch.
    let switchLlmCalled = false;
    let switchLlmBody: Record<string, unknown> | null = null;
    page.on("request", (req) => {
      const url = new URL(req.url());
      if (
        req.method() === "POST" &&
        url.pathname.match(/\/api\/conversations\/[^/]+\/switch_llm/)
      ) {
        switchLlmCalled = true;
        try {
          switchLlmBody = req.postDataJSON();
        } catch {
          // non-JSON body
        }
      }
    });

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    // ── Send initial message and wait for agent reply ──

    await test.step("send initial message", async () => {
      await setChatInput(page, "Hello, please respond briefly.");
      await page.getByTestId("submit-button").click();
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    journey.conversationIds.add(conversationId);

    await test.step("wait for initial agent reply", async () => {
      await waitForNonUserMessageText(page, INITIAL_REPLY_TOKEN, 30_000);
    });

    // ── Type /model <profile-B> to switch ──

    await test.step("type /model command to switch to profile B", async () => {
      // Wait for the chat input to be available (it should be focused
      // after the agent reply completes).
      await waitForTestId(page, "chat-input");

      await setChatInput(page, `/model ${PROFILE_B_NAME}`);
      await page.getByTestId("submit-button").click();
    });

    // ── Verify: "Switched to profile" message appears in chat UI ──

    await test.step("verify 'Switched to profile' message in chat", async () => {
      // waitForNonUserMessageText already polls data-testid="model-messages"
      // elements, so a successful wait proves the container is visible.
      await waitForNonUserMessageText(page, PROFILE_B_NAME, 30_000);
    });

    // ── Verify: the switch_llm POST was made ──

    await test.step("verify switch_llm API was called with profile B model", async () => {
      expect(switchLlmCalled, "POST /switch_llm should have been called").toBe(
        true,
      );
      expect(switchLlmBody).toBeTruthy();
      // ConversationClient.switchLLM posts { llm: <config> } to switch_llm,
      // so the model is nested under the "llm" key in the HTTP body.
      const llm = switchLlmBody!.llm as Record<string, unknown> | undefined;
      expect(llm, "switch_llm body should contain an llm object").toBeTruthy();
      expect(
        llm!.model,
        `switch_llm body.llm.model should be "${MODEL_B}"`,
      ).toBe(MODEL_B);
    });

    // ── Send a follow-up message to verify conversation still works ──

    await test.step("send follow-up message after switch", async () => {
      // Wait for the chat input to be ready — the UI may briefly disable
      // it while the profile switch settles.
      await waitForTestId(page, "chat-input");
      await setChatInput(page, "Confirm the model switch worked.");
      await page.getByTestId("submit-button").click();
    });

    await test.step("verify post-switch agent reply", async () => {
      await waitForNonUserMessageText(page, POST_SWITCH_REPLY_TOKEN, 30_000);
    });

    // ── Verify: no error banners ──

    await test.step("verify no error banners", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
