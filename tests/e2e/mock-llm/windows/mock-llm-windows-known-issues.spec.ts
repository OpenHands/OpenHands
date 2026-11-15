/**
 * Mock-LLM E2E: behaviour that is known to break on a Windows host.
 *
 * Every test asserts the correct, platform-independent behaviour, so it
 * passes on Linux and macOS and fails on Windows until its issue is fixed.
 * The Windows npm-install workflow runs this file as its own job
 * (`known-windows-bugs` in .github/workflows/windows-install-e2e.yml): a fix
 * PR shows its test going from red to green there.
 *
 * Tests are independent (no serial mode), so one failure does not skip the
 * rest. Commands given to the agent must run in both bash and PowerShell (the
 * agent-server's Windows terminal), so they only use `mkdir`, `echo` and `>`.
 */

import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForAgentMessageContaining,
  deleteConversation,
  ensureMockLLMProfile,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  setChatInput,
} from "../utils/mock-llm-helpers";

const REPLY_TOKEN = "WINDOWS_KNOWN_ISSUES_REPLY_OK";
const FILE_NAME = "windows-check.txt";
const PLAN_HEADING = "Windows plan check";
const WORKSPACE_COMMAND = [
  "mkdir .agents_tmp",
  `echo "# ${PLAN_HEADING}" > .agents_tmp/PLAN.md`,
  `echo windows-check > ${FILE_NAME}`,
].join("; ");
const AUTOMATION_API_BASE = `${BACKEND_URL}/api/automation/v1`;
const headers = { "X-Session-API-Key": SESSION_API_KEY };

/** Wait for the agent's observation from `toolName` and return it. */
async function waitForToolObservation(
  request: APIRequestContext,
  conversationId: string,
  toolName: string,
  timeout: number,
) {
  let found: any;
  await expect
    .poll(
      async () => {
        const resp = await request.get(
          `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
          { headers, params: { limit: "100", sort_order: "TIMESTAMP_DESC" } },
        );
        if (!resp.ok()) return false;
        const items: any[] = (await resp.json()).items ?? [];
        found = items.find(
          (e) => e.kind === "ObservationEvent" && e.tool_name === toolName,
        );
        return Boolean(found);
      },
      { timeout, message: `No ${toolName} observation in ${conversationId}` },
    )
    .toBe(true);
  return found.observation as {
    is_error?: boolean;
    content?: { text?: string }[];
  };
}

function observationText(observation: { content?: { text?: string }[] }) {
  return (observation.content ?? []).map((c) => c.text ?? "").join("");
}

const conversationIds = new Set<string>();

async function startConversation(page: Page, message: string) {
  await routeSessionApiKey(page);
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "home-chat-launcher");
  await setChatInput(page, message);
  await page.getByTestId("submit-button").click();
  await waitForPath(page, /\/conversations\/.+/, 30_000);
  const id = getConversationIdFromURL(page);
  conversationIds.add(id);
  return id;
}

/**
 * Start a conversation whose agent writes FILE_NAME and .agents_tmp/PLAN.md
 * into its workspace, and wait until the command has run.
 */
async function createWorkspaceConversation(
  page: Page,
  request: APIRequestContext,
) {
  await ensureMockLLMProfile(page);
  await resetMockLLM(request);
  // Turn 0 pads the title-generation call (see mock-llm-conversation.spec.ts).
  await registerTrajectory(request, "windows-workspace", [
    { text: "" },
    {
      tool_call: {
        name: "terminal",
        arguments: { command: WORKSPACE_COMMAND },
      },
    },
    { text: REPLY_TOKEN },
  ]);
  await activateTrajectory(request, "windows-workspace");

  const id = await startConversation(page, "Create a file and a plan.");
  const observation = await waitForToolObservation(
    request,
    id,
    "terminal",
    90_000,
  );
  expect(observation.is_error, observationText(observation)).toBeFalsy();
  await waitForAgentMessageContaining(request, id, REPLY_TOKEN, 60_000);
  return id;
}

async function openConversationPanel(page: Page, conversationId: string) {
  await routeSessionApiKey(page);
  await page.goto(`/conversations/${conversationId}`, {
    waitUntil: "domcontentloaded",
  });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "chat-interface", 30_000);
  const toggle = page.getByTestId("right-panel-toggle");
  await expect(toggle).toBeVisible({ timeout: 10_000 });
  await toggle.click();
  await expect(page.getByTestId("conversation-tab-files")).toBeVisible({
    timeout: 10_000,
  });
}

test.describe("Windows known issues", () => {
  // The bugs are deterministic: retrying a failure only doubles the time.
  test.describe.configure({ retries: 0 });

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    for (const id of conversationIds) {
      await deleteConversation(request, id).catch(() => {});
      conversationIds.delete(id);
    }
    await resetMockLLM(request).catch(() => {});
  });

  test("Files tab lists files the agent created (#17773)", async ({
    page,
    request,
  }) => {
    test.setTimeout(240_000);
    const id = await createWorkspaceConversation(page, request);

    await openConversationPanel(page, id);
    if (!(await page.getByTestId("files-tab").isVisible())) {
      await page.getByTestId("conversation-tab-files").click();
    }
    await expect(page.getByTestId("files-tab")).toBeVisible({
      timeout: 15_000,
    });
    // The file already exists when the page loads, so the tab's first
    // listing must include it.
    await expect(page.getByTestId(`file-tree-file-${FILE_NAME}`)).toBeVisible({
      timeout: 20_000,
    });
  });

  test("Planner tab shows the workspace PLAN.md (#17774)", async ({
    page,
    request,
  }) => {
    test.setTimeout(240_000);
    const id = await createWorkspaceConversation(page, request);

    await openConversationPanel(page, id);
    const plannerTab = page.getByTestId("conversation-tab-planner");
    if (await plannerTab.isVisible()) {
      await plannerTab.click();
    } else {
      await page
        .getByTestId("tabs-pane-header")
        .getByTestId("ellipsis-button")
        .click();
      await page.getByTestId("conversation-tabs-menu-open-planner").click();
    }
    await expect(page.getByRole("heading", { name: PLAN_HEADING })).toBeVisible(
      { timeout: 20_000 },
    );
  });

  test("an automation run completes (#15509)", async ({ page, request }) => {
    test.setTimeout(300_000);
    await ensureMockLLMProfile(page);

    // The run's conversation: padding, the finish tool that
    // openhands-automation requires, a final reply and a safety blank
    // (same turns as mock-llm-automation.spec.ts).
    const runTurns = [
      { text: "" },
      {
        tool_call: {
          name: "finish",
          arguments: {
            message: "Windows automation run finished.",
            status: "success",
            outcome_summary: "Windows automation run finished.",
          },
        },
      },
      { text: "Done." },
      { text: "" },
    ];
    await resetMockLLM(request);
    await registerTrajectory(request, "windows-automation", [
      ...runTurns,
      ...runTurns, // one re-dispatch, see below
    ]);
    await activateTrajectory(request, "windows-automation");

    const created = await request.post(`${AUTOMATION_API_BASE}/preset/prompt`, {
      headers,
      data: {
        name: "Windows known issues",
        prompt: "Say hello.",
        trigger: { type: "cron", schedule: "0 9 * * *", timezone: "UTC" },
      },
    });
    expect(created.ok(), `create automation: ${created.status()}`).toBe(true);
    const automationId: string = (await created.json()).id;

    try {
      // Re-dispatch once after a FAILED run, like mock-llm-automation.spec.ts:
      // the completion callback is intermittently flaky on every platform.
      let runs: { status: string; error_detail?: string | null }[] = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        const dispatched = await request.post(
          `${AUTOMATION_API_BASE}/${automationId}/dispatch`,
          { headers },
        );
        expect(dispatched.ok(), `dispatch: ${dispatched.status()}`).toBe(true);
        await expect
          .poll(
            async () => {
              const resp = await request.get(
                `${AUTOMATION_API_BASE}/${automationId}/runs`,
                { headers },
              );
              if (!resp.ok()) return false;
              const data = await resp.json();
              runs = data.runs ?? data.items ?? [];
              return (
                runs.length > attempt &&
                runs.every((r) =>
                  ["COMPLETED", "FAILED", "CANCELLED", "SKIPPED"].includes(
                    r.status,
                  ),
                )
              );
            },
            { timeout: 90_000, message: "automation run did not finish" },
          )
          .toBe(true);
        if (runs.some((r) => r.status === "COMPLETED")) break;
      }
      const summary = runs
        .map(
          (r) => `${r.status}${r.error_detail ? ` (${r.error_detail})` : ""}`,
        )
        .join(", ");
      expect(
        runs.some((r) => r.status === "COMPLETED"),
        `runs: ${summary}`,
      ).toBe(true);
    } finally {
      await request
        .delete(`${AUTOMATION_API_BASE}/${automationId}`, { headers })
        .catch(() => {});
    }
  });

  test("the browser tool can open a page (#16551)", async ({
    page,
    request,
  }) => {
    // The mock-LLM configs start the stack with browser tools off; the
    // Windows npm-install job leaves them on and sets this variable.
    test.skip(
      process.env.MOCK_LLM_BROWSER_TOOLS !== "1",
      "browser tools are disabled for this stack",
    );
    // #16551: the Windows workflow blocks remote debugging in system Chrome,
    // which the SDK launches before Playwright's Chromium, so this fails
    // there until the SDK prefers Playwright's Chromium.
    test.setTimeout(240_000);
    await ensureMockLLMProfile(page);

    await resetMockLLM(request);
    await registerTrajectory(request, "windows-browser", [
      { text: "" },
      {
        tool_call: {
          name: "browser_navigate",
          arguments: { url: `${BACKEND_URL}/`, new_tab: false },
        },
      },
      { text: REPLY_TOKEN },
    ]);
    await activateTrajectory(request, "windows-browser");

    const id = await startConversation(page, "Open the Canvas home page.");
    const observation = await waitForToolObservation(
      request,
      id,
      "browser_navigate",
      180_000,
    );
    expect(observation.is_error, observationText(observation)).toBeFalsy();
  });
});
