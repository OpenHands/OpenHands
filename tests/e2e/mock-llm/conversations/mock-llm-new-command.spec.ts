/**
 * Mock-LLM E2E coverage for local `/new` workspace reuse.
 *
 * The source agent creates an untracked file. `/new` must attach the exact
 * same working directory (without asking for another worktree), after which a
 * second agent command proves that the uncommitted file is visible.
 */

import { expect, test, type APIRequestContext } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  activateTrajectory,
  deleteConversation,
  dismissAnalyticsModal,
  ensureMockLLMProfile,
  getConversationIdFromURL,
  registerTrajectory,
  resetMockLLM,
  routeSessionApiKey,
  seedLocalStorage,
  setChatInput,
  waitForNonUserMessageText,
  waitForPath,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const PROFILE_NAME = "mock-llm-new-command";
const FILE_NAME = ".canvas-new-uncommitted.txt";
const FILE_CONTENT = "LOCAL_NEW_REUSES_UNCOMMITTED_WORKSPACE";
const SOURCE_READY_TOKEN = "LOCAL_NEW_SOURCE_READY";
const REUSE_PROOF_TOKEN = "LOCAL_NEW_REUSE_PROVED";
const NEW_READY_TOKEN = "LOCAL_NEW_DESTINATION_READY";
const DESTINATION_COMMAND =
  `node -e "const fs=require('node:fs');` +
  `const value=fs.readFileSync('${FILE_NAME}','utf8').trim();` +
  `if(value!=='${FILE_CONTENT}')process.exit(1);` +
  `console.log('${REUSE_PROOF_TOKEN}')"`;

async function waitForSuccessfulBashCommand(
  request: APIRequestContext,
  conversationId: string,
  command: string,
) {
  let lastEvents: unknown[] = [];
  await expect
    .poll(
      async () => {
        const response = await request.get(
          `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
          {
            headers: { "X-Session-API-Key": SESSION_API_KEY },
            params: { limit: "100", sort_order: "TIMESTAMP_DESC" },
          },
        );
        if (!response.ok()) return false;
        const body = (await response.json()) as { items?: unknown[] };
        lastEvents = body.items ?? [];
        return lastEvents.some((event) => {
          if (typeof event !== "object" || event === null) return false;
          const observation = (
            event as { observation?: Record<string, unknown> }
          ).observation;
          if (!observation) return false;
          const isTerminalObservation =
            observation.kind === "ExecuteBashObservation" ||
            observation.kind === "TerminalObservation";
          return (
            isTerminalObservation &&
            observation.command === command &&
            observation.exit_code === 0 &&
            observation.error !== true &&
            observation.is_error !== true &&
            observation.timeout !== true
          );
        });
      },
      { timeout: 60_000 },
    )
    .toBe(true)
    .catch((error) => {
      throw new Error(
        `No successful matching terminal observation. Last events: ${JSON.stringify(lastEvents)}`,
        { cause: error },
      );
    });
}

test("local /new reuses the exact workspace and sees an uncommitted file", async ({
  page,
  request,
}) => {
  test.setTimeout(180_000);
  const conversationIds = new Set<string>();
  const creationPayloads: Record<string, unknown>[] = [];

  await seedLocalStorage(page);
  await ensureMockLLMProfile(page, { profileName: PROFILE_NAME });
  await registerTrajectory(request, "local-new-source", [
    { text: "" },
    {
      tool_call: {
        name: "terminal",
        arguments: {
          command:
            `node -e "require('node:fs').writeFileSync(` +
            `'${FILE_NAME}','${FILE_CONTENT}\\n')"`,
        },
      },
    },
    { text: SOURCE_READY_TOKEN },
  ]);
  await activateTrajectory(request, "local-new-source");

  const captureCreationPayload = (
    requestEvent: import("@playwright/test").Request,
  ) => {
    if (
      requestEvent.method() === "POST" &&
      new URL(requestEvent.url()).pathname === "/api/conversations"
    ) {
      creationPayloads.push(requestEvent.postDataJSON());
    }
  };
  page.on("request", captureCreationPayload);

  try {
    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");
    await setChatInput(page, "Create the local workspace marker.");
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    const sourceConversationId = getConversationIdFromURL(page);
    conversationIds.add(sourceConversationId);
    await waitForNonUserMessageText(page, SOURCE_READY_TOKEN, 60_000);

    await registerTrajectory(request, "local-new-destination", [
      { text: "" },
      {
        tool_call: {
          name: "terminal",
          arguments: { command: DESTINATION_COMMAND },
        },
      },
      { text: NEW_READY_TOKEN },
    ]);
    await activateTrajectory(request, "local-new-destination");

    await setChatInput(page, "/new");
    await page.getByTestId("submit-button").click();
    await expect
      .poll(() => getConversationIdFromURL(page), { timeout: 30_000 })
      .not.toBe(sourceConversationId);
    const destinationConversationId = getConversationIdFromURL(page);
    conversationIds.add(destinationConversationId);
    await waitForTestId(page, "chat-interface", 30_000);

    expect(creationPayloads).toHaveLength(2);
    const sourceWorkspace = creationPayloads[0].workspace as {
      working_dir?: string;
    };
    const destinationWorkspace = creationPayloads[1].workspace as {
      working_dir?: string;
    };
    expect(destinationWorkspace.working_dir).toBe(sourceWorkspace.working_dir);
    expect(creationPayloads[1].worktree).toBe(false);

    await setChatInput(page, "Verify the workspace marker.");
    await page.getByTestId("submit-button").click();
    await waitForNonUserMessageText(page, NEW_READY_TOKEN, 60_000);
    await waitForSuccessfulBashCommand(
      request,
      destinationConversationId,
      DESTINATION_COMMAND,
    );
  } finally {
    page.off("request", captureCreationPayload);
    for (const conversationId of conversationIds) {
      await deleteConversation(request, conversationId).catch(() => undefined);
    }
    await resetMockLLM(request);
  }
});
