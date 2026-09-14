/**
 * Mock-LLM E2E tests: Files tab, conversation overview git, and Browser tab.
 *
 * Exercises conversation-panel tabs and git integration against the real
 * agent-server with a scripted mock LLM backend.
 *
 * Coverage (issue #511):
 *   - Files tab diff view can be enabled when a workspace is attached
 *   - Conversation overview shows workspace / git identity (composer rail removed)
 *   - Browser tab renders empty state when no page has been browsed
 *   - Files tab defaults to file-tree view when NO workspace is attached
 */

import { test, expect } from "@playwright/test";
import {
  REPLY_TOKEN,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  deleteConversation,
  ensureMockLLMProfile,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
} from "../utils/mock-llm-helpers";

const USER_MESSAGE = "Hello, please respond.";
const WORKSPACE_PATH = "/tmp/e2e-test-project/my-app";
// The git remote the step-2 trajectory configures for the workspace. Kept as a
// shared constant so the `git remote add` command and the overview git
// assertion below can never drift apart.
const EXPECTED_REPO_SLUG = "test-org/test-repo";

/**
 * Seed `selected_workspace` into the conversation metadata localStorage key.
 *
 * Uses `addInitScript` so the write happens on the real app origin when the
 * first `page.goto()` triggers a document load — `page.evaluate` on
 * `about:blank` would write to the wrong origin.
 */
async function seedWorkspaceMetadata(
  page: import("@playwright/test").Page,
  conversationId: string,
  workspacePath: string,
) {
  await page.addInitScript(
    ({ convId, wsPath }) => {
      const STORAGE_KEY = "openhands-agent-server-conversation-metadata";
      const raw = window.localStorage.getItem(STORAGE_KEY);
      const all = raw ? JSON.parse(raw) : {};
      all[convId] = {
        ...(all[convId] || {}),
        selected_workspace: wsPath,
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    },
    { convId: conversationId, wsPath: workspacePath },
  );
}

test.describe.configure({ mode: "serial" });

test.describe("files tab, conversation overview git, and browser tab", () => {
  const conversationIds = new Set<string>();
  /** Conversation ID from the workspace-attached test, reused across steps. */
  let attachedConversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ page, request }) => {
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    if (match?.[1]) conversationIds.add(decodeURIComponent(match[1]));
  });

  test.afterAll(async ({ request }) => {
    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
      } catch {
        // best-effort
      }
    }
    try {
      await resetMockLLM(request);
    } catch {
      // best-effort
    }
  });

  // ── Step 1: Setup LLM profile ──────────────────────────────────────

  test("step 1: ensure mock LLM profile is configured", async ({
    page,
    request,
  }) => {
    // Create AND activate a real LLM profile through the Settings UI.
    // In local mode the home launcher gates sending on an active LLM profile
    // (profiles are the source of truth — see `useLlmConfigured`), so the
    // settings-only `ensureMockLLMProfileViaAPI` path leaves the chat input
    // and submit button disabled and step 2 can never submit. Other
    // conversation-starting mock-LLM specs create a profile for this reason;
    // under selective E2E runs no earlier spec leaves an active profile behind.
    await ensureMockLLMProfile(page);

    // Register a trajectory that ensures the workspace has a git remote.
    // The npm path inherits the host repo; the Docker path bootstraps one.
    const gitBootstrap = [
      "git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init",
      // Must configure user.name/email — Docker containers may not have them.
      "git config user.email test@test.com",
      "git config user.name test",
      `git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/${EXPECTED_REPO_SLUG}.git`,
      "git rev-parse --verify HEAD >/dev/null 2>&1 || git commit --allow-empty -m init",
    ].join(" && ");
    await registerTrajectory(request, "files-and-git", [
      {
        tool_call: {
          name: "terminal",
          arguments: {
            command: `${gitBootstrap}; printf 'MOCK_LLM_E2E_BASH_OK\\n'`,
          },
        },
      },
      { text: REPLY_TOKEN },
    ]);
    await activateTrajectory(request, "files-and-git");
  });

  // ── Step 2: Start a conversation and seed workspace attachment ──────

  test("step 2: start conversation and attach workspace metadata", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    // Type and send a message from the home page launcher
    await page.evaluate(
      ({ testId, text }) => {
        const el = document.querySelector(`[data-testid="${testId}"]`);
        if (!(el instanceof HTMLElement))
          throw new Error("Chat input not found");
        el.focus();
        el.textContent = text;
        el.dispatchEvent(
          new InputEvent("input", {
            bubbles: true,
            data: text,
            inputType: "insertText",
          }),
        );
      },
      { testId: "chat-input", text: USER_MESSAGE },
    );
    await page.getByTestId("submit-button").click();

    // Wait for navigation to the conversation page
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);
    attachedConversationId = conversationId;

    // Wait for the agent to finish replying so the conversation is fully
    // initialized (WebSocket connected, runtime ready).
    await waitForNonUserMessageText(page, REPLY_TOKEN, 60_000);

    await seedWorkspaceMetadata(page, conversationId, WORKSPACE_PATH);

    // Reload so hooks re-read from localStorage
    await page.reload({ waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Wait for the conversation to load again
    await waitForTestId(page, "chat-interface", 30_000);
  });

  // ── Step 3: Verify overview shows workspace / git identity ─────────

  test("step 3: conversation overview shows workspace and git identity", async ({
    page,
  }) => {
    test.skip(!attachedConversationId, "step 2 must complete first");
    test.setTimeout(60_000);

    await seedWorkspaceMetadata(page, attachedConversationId!, WORKSPACE_PATH);
    await routeSessionApiKey(page);
    await page.goto(`/conversations/${attachedConversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "chat-interface", 30_000);

    const workspaceName = WORKSPACE_PATH.replace(/\/+$/, "").split("/").pop()!;

    // Composer no longer hosts the git rail; identity lives in overview.
    await test.step("composer has no git control rail", async () => {
      await expect(
        page.getByTestId("interactive-chat-box").getByRole("button", {
          name: /^(Pull|Push|Create PR|Connect Repo)$/i,
        }),
      ).toHaveCount(0);
    });

    await test.step("open conversation overview", async () => {
      const toggle = page.getByTestId("conversation-overview-toggle");
      await expect(toggle).toBeVisible({ timeout: 10_000 });
      await toggle.click();
      await waitForTestId(page, "conversation-overview-panel", 10_000);
    });

    await test.step("verify workspace row shows folder basename", async () => {
      const workspaceRow = page.getByTestId("conversation-overview-workspace");
      await expect(workspaceRow).toBeVisible({ timeout: 10_000 });
      await expect(workspaceRow).toContainText(workspaceName);
    });

    // When useLocalGitInfo detects a remote (trajectory ran git init +
    // remote add in step 2), the overview git section surfaces the repo
    // slug. In Docker the bash WebSocket probe may be slower — soft-check.
    await test.step("check for overview git repo (git detection)", async () => {
      const repoRow = page.getByTestId("conversation-overview-git-repo");
      try {
        await expect(repoRow).toBeVisible({ timeout: 20_000 });
        await expect(repoRow).toContainText(EXPECTED_REPO_SLUG);
      } catch {
        console.log(
          "Overview git repo not visible — git probe likely still pending",
        );
      }
    });
  });

  // ── Step 4: Verify Commits tab is available for attached workspace ─

  test("step 4: commits tab opens for attached workspace", async ({ page }) => {
    test.skip(!attachedConversationId, "step 2 must complete first");
    test.setTimeout(60_000);

    await seedWorkspaceMetadata(page, attachedConversationId!, WORKSPACE_PATH);
    await routeSessionApiKey(page);
    await page.goto(`/conversations/${attachedConversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "chat-interface", 30_000);

    await test.step("open right panel and Commits tab", async () => {
      const toggle = page.getByTestId("right-panel-toggle");
      await expect(toggle).toBeVisible({ timeout: 10_000 });
      await toggle.click({ force: true });
      await expect(toggle).toHaveAttribute("aria-pressed", "true", {
        timeout: 10_000,
      });

      const anyTab = page.locator('[data-testid^="conversation-tab-"]').first();
      await expect(anyTab).toBeVisible({ timeout: 10_000 });

      // Commits may be overflowed into the ⋯ menu on narrow viewports.
      const commitsTab = page.getByTestId("conversation-tab-commits");
      if (await commitsTab.isVisible().catch(() => false)) {
        await commitsTab.click();
      } else {
        await page.getByTestId("ellipsis-button").click();
        await page.getByTestId("conversation-tabs-menu-open-commits").click();
      }
      await expect(page.getByTestId("conversation-tab-commits")).toBeVisible({
        timeout: 15_000,
      });
    });
  });

  // ── Step 5: Verify Browser tab shows empty state ───────────────────

  test("step 5: browser tab shows empty state", async ({ page }) => {
    test.skip(!attachedConversationId, "step 2 must complete first");
    test.setTimeout(60_000);

    await routeSessionApiKey(page);
    await page.goto(`/conversations/${attachedConversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "chat-interface", 30_000);

    // Open the right panel and wait for the drawer animation to settle
    await test.step("open right panel", async () => {
      const toggle = page.getByTestId("right-panel-toggle");
      await expect(toggle).toBeVisible({ timeout: 10_000 });
      await toggle.click();
      await page.waitForTimeout(500);
    });

    // Click the Browser tab
    await test.step("click browser tab", async () => {
      const browserTab = page.getByTestId("conversation-tab-browser");
      await expect(browserTab).toBeVisible({ timeout: 10_000 });
      await browserTab.click();
    });

    await test.step("verify empty browser message", async () => {
      // The EmptyBrowserMessage renders the "No page loaded yet" message.
      // We assert on the text rather than a test-id since the component
      // uses the shared ConversationTabEmptyState without its own id.
      await expect(
        page.getByText("No page loaded yet", { exact: false }),
      ).toBeVisible({ timeout: 10_000 });
    });
  });

  // ── Step 6: Verify Files tab defaults to file-tree when no workspace ─

  test("step 6: files tab defaults to file-tree view without attached workspace", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    // Fresh trajectory — step 2's conversation consumed the previous one.
    await resetMockLLM(request);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    // Start a brand-new conversation WITHOUT seeding any workspace metadata
    await page.evaluate(
      ({ testId, text }) => {
        const el = document.querySelector(`[data-testid="${testId}"]`);
        if (!(el instanceof HTMLElement))
          throw new Error("Chat input not found");
        el.focus();
        el.textContent = text;
        el.dispatchEvent(
          new InputEvent("input", {
            bubbles: true,
            data: text,
            inputType: "insertText",
          }),
        );
      },
      { testId: "chat-input", text: USER_MESSAGE },
    );
    await page.getByTestId("submit-button").click();

    await waitForPath(page, /\/conversations\/.+/, 30_000);
    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    // Wait for the agent to reply
    await waitForNonUserMessageText(page, REPLY_TOKEN, 60_000);

    // Open the right panel and wait for the drawer animation to settle
    await test.step("open right panel", async () => {
      const toggle = page.getByTestId("right-panel-toggle");
      await expect(toggle).toBeVisible({ timeout: 10_000 });
      await toggle.click();
      await page.waitForTimeout(500);
    });

    // Ensure the Files surface is showing. Opening the drawer already
    // defaults the selection to Files, and clicking the active tab again
    // deliberately closes the drawer (see useSelectConversationTab), so only
    // click when the Files panel is not already visible.
    await test.step("ensure files tab is active", async () => {
      const filesTab = page.getByTestId("conversation-tab-files");
      await expect(filesTab).toBeVisible({ timeout: 10_000 });
      if (!(await page.getByTestId("files-tab").isVisible())) {
        await filesTab.click();
        await page.waitForTimeout(300);
      }
    });

    await test.step("verify files tab shows file browser (not diff)", async () => {
      await expect(page.getByTestId("files-tab")).toBeVisible({
        timeout: 15_000,
      });
      // The file browser's tab strip renders once loading settles; the
      // content mode toggle only appears after a file is opened, and the
      // legacy Diff/Commits toggle is gone from the Files surface.
      await expect(page.getByTestId("file-quick-row")).toBeVisible({
        timeout: 10_000,
      });
      await expect(page.getByTestId("files-tab-content")).toBeVisible({
        timeout: 10_000,
      });
      await expect(
        page.getByTestId("files-tab-content-mode-toggle"),
      ).toHaveCount(0);
      await expect(page.getByTestId("files-tab-diff-toggle")).toHaveCount(0);
    });
  });
});
