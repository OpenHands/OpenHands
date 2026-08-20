import { test, type APIRequestContext } from "@playwright/test";

import {
  BACKEND_URL,
  clickButtonByTestId,
  configureLiveAgentServer,
  createLiveConversation,
  deleteLiveLlmProfile,
  dismissAnalyticsModal,
  enableLiveE2EFlags,
  EXPECTED_BASH_COMMAND,
  EXPECTED_BASH_OUTPUT_TOKEN,
  EXPECTED_REPLY_TOKEN,
  expandVisibleEventDetails,
  fillChatInput,
  getLiveArtifactMask,
  getOptionalConversationIdFromURL,
  guardAgainstPostHogRequests,
  hasLiveLLMConfig,
  missingLiveLLMConfigMessage,
  routeBackendSessionApiKey,
  sessionApiKey,
  waitForAgentReply,
  waitForCriticResultDisplay,
  waitForCriticResultEvent,
  waitForNonUserMessageText,
  waitForSuccessfulBashObservation,
  waitForTestId,
} from "./utils/agent-server-conversation";

test.describe("live Agent Server terminal conversation", () => {
  const createdConversationIds = new Set<string>();

  async function deleteConversation(
    request: APIRequestContext,
    conversationId: string,
  ) {
    const response = await request.delete(
      `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}`,
      {
        headers: {
          "X-Session-API-Key": sessionApiKey,
        },
      },
    );
    if (response.ok() || response.status() === 404) {
      createdConversationIds.delete(conversationId);
      return;
    }

    throw new Error(
      `Failed to clean up live E2E conversation ${conversationId}: ${response.status()}`,
    );
  }

  async function cleanupKnownConversations(request: APIRequestContext) {
    const cleanupErrors: string[] = [];

    for (const conversationId of Array.from(createdConversationIds)) {
      try {
        await deleteConversation(request, conversationId);
      } catch (error) {
        cleanupErrors.push(
          error instanceof Error ? error.message : String(error),
        );
      }
    }

    if (cleanupErrors.length > 0) {
      throw new Error(cleanupErrors.join("\n"));
    }
  }

  test.beforeEach(async ({ page }) => {
    await enableLiveE2EFlags(page);
  });

  test.afterEach(async ({ page, request }) => {
    const conversationId = getOptionalConversationIdFromURL(page);
    if (conversationId) {
      createdConversationIds.add(conversationId);
    }

    await cleanupKnownConversations(request);
    await deleteLiveLlmProfile(request);
  });

  test.afterAll(async ({ request }) => {
    await cleanupKnownConversations(request);
    await deleteLiveLlmProfile(request);
  });

  test("runs a real LLM-backed Agent Server terminal conversation through the UI", async ({
    page,
    request,
  }, testInfo) => {
    test.skip(!hasLiveLLMConfig, missingLiveLLMConfigMessage);

    await configureLiveAgentServer(request);
    const conversationId = await createLiveConversation(request);
    createdConversationIds.add(conversationId);
    await routeBackendSessionApiKey(page);
    const postHogGuard = await guardAgainstPostHogRequests(page);

    await page.goto(`/conversations/${conversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "app-route");
    await waitForTestId(page, "chat-interface");
    await waitForTestId(page, "interactive-chat-box");

    await fillChatInput(
      page,
      [
        "Use the terminal/bash tool exactly once.",
        `Run this exact command: ${EXPECTED_BASH_COMMAND}`,
        `After the command succeeds, reply with exactly this token and then finish: ${EXPECTED_REPLY_TOKEN}`,
        "Do not use any other tools. Do not add any other text in the final reply.",
      ].join("\n"),
    );
    await clickButtonByTestId(page, "submit-button");

    await waitForAgentReply(page);
    await waitForSuccessfulBashObservation(request, conversationId);
    await expandVisibleEventDetails(page);
    await waitForNonUserMessageText(page, EXPECTED_BASH_OUTPUT_TOKEN);

    const screenshotPath = testInfo.outputPath("live-agent-response.png");
    await page.getByTestId("chat-interface").screenshot({
      path: screenshotPath,
      mask: getLiveArtifactMask(page),
    });
    await testInfo.attach("live-agent-response", {
      path: screenshotPath,
      contentType: "image/png",
    });

    await postHogGuard.expectNoRequests();
  });

  test("renders critic evaluation results for a real LLM-backed conversation", async ({
    page,
    request,
  }, testInfo) => {
    test.skip(!hasLiveLLMConfig, missingLiveLLMConfigMessage);
    test.setTimeout(240_000);

    await configureLiveAgentServer(request, { enableCritic: true });
    const conversationId = await createLiveConversation(request, {
      enableCritic: true,
    });
    createdConversationIds.add(conversationId);
    await routeBackendSessionApiKey(page);
    const postHogGuard = await guardAgainstPostHogRequests(page);

    await page.goto(`/conversations/${conversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "app-route");
    await waitForTestId(page, "chat-interface");
    await waitForTestId(page, "interactive-chat-box");

    await fillChatInput(
      page,
      [
        "Use the terminal/bash tool exactly once.",
        `Run this exact command: ${EXPECTED_BASH_COMMAND}`,
        `After the command succeeds, reply with exactly this token and then finish: ${EXPECTED_REPLY_TOKEN}`,
        "Do not use any other tools. Do not add any other text in the final reply.",
      ].join("\n"),
    );
    await clickButtonByTestId(page, "submit-button");

    await waitForAgentReply(page);
    await waitForSuccessfulBashObservation(request, conversationId);
    await waitForCriticResultEvent(request, conversationId);
    await waitForCriticResultDisplay(page);

    const screenshotPath = testInfo.outputPath("live-critic-result.png");
    await page.getByTestId("chat-interface").screenshot({
      path: screenshotPath,
      mask: getLiveArtifactMask(page),
    });
    await testInfo.attach("live-critic-result", {
      path: screenshotPath,
      contentType: "image/png",
    });

    await postHogGuard.expectNoRequests();
  });
});
