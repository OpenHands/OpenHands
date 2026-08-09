import { expect, test } from "@playwright/test";
import {
  activateTrajectory,
  deleteConversation,
  dismissAnalyticsModal,
  ensureMockLLMProfile,
  getConversationIdFromURL,
  getMockLLMRequests,
  registerTrajectory,
  resetMockLLM,
  routeSessionApiKey,
  seedLocalStorage,
  setChatInput,
  waitForNonUserMessageText,
  waitForPath,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const INITIAL_REPLY = "MOCK_LLM_FILE_UPLOAD_READY";
const UPLOAD_REPLY = "MOCK_LLM_FILE_UPLOAD_DONE";
const FILE_NAME = "bug.txt";

test("an uploaded file reaches the agent by absolute path and appears without a manual refresh", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  let conversationId: string | null = null;

  try {
    await seedLocalStorage(page);
    await ensureMockLLMProfile(page);
    await resetMockLLM(request);
    await registerTrajectory(request, "file-upload-path", [
      { text: "" },
      { text: INITIAL_REPLY },
      { text: UPLOAD_REPLY },
    ]);
    await activateTrajectory(request, "file-upload-path");

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    await setChatInput(page, "Start a conversation for the upload test.");
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    conversationId = getConversationIdFromURL(page);
    await waitForNonUserMessageText(page, INITIAL_REPLY, 30_000);

    const panelToggle = page.getByTestId("right-panel-toggle");
    await panelToggle.click();
    await expect(panelToggle).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("conversation-tab-files").click();
    await expect(page.getByTestId("files-tab-diff-toggle")).toBeVisible();
    await expect(
      page.getByTestId("files-tab-diff-toggle-option-off"),
    ).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("file-quick-row-tree-toggle").click();
    await expect(page.getByTestId("files-tab-tree")).toBeVisible();

    await page.getByTestId("upload-image-input").setInputFiles({
      name: FILE_NAME,
      mimeType: "text/plain",
      buffer: Buffer.from("uploaded file contents"),
    });
    await setChatInput(page, "Read the uploaded file.");
    await expect(page.getByTestId("submit-button")).toBeEnabled();
    await page.getByTestId("submit-button").click();

    await waitForNonUserMessageText(page, UPLOAD_REPLY, 30_000);

    await expect
      .poll(
        async () => {
          const requests = await getMockLLMRequests(request);
          return requests.some((body) => {
            const serialized = JSON.stringify(body);
            return new RegExp(`(?:/[^"\\s]+)+/${FILE_NAME}`).test(serialized);
          });
        },
        { timeout: 15_000 },
      )
      .toBe(true);

    await expect(page.getByTestId(`file-tree-file-${FILE_NAME}`)).toBeVisible({
      timeout: 15_000,
    });
  } finally {
    if (conversationId) {
      await deleteConversation(request, conversationId).catch(() => undefined);
    }
    await resetMockLLM(request).catch(() => undefined);
  }
});
