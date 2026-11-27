/**
 * Mock-LLM E2E test: folder browsing → workspace selection → conversation creation.
 *
 * Covers two "I can" statements from issue #511:
 *   - "I can browse local files and folders to choose where to begin"
 *   - "I can start a conversation against a local Git repo without typing the path"
 *
 * Flow (serial):
 *   1. Open the folder browser, navigate to a known test directory, click
 *      "Use this folder" — verify the workspace is auto-selected
 *   2. Confirm the workspace, type a message, submit — intercept
 *      POST /api/conversations and assert workspace.working_dir matches
 *      the selected folder path
 *   3. After conversation creation, verify selected_workspace is persisted
 *      in localStorage under the conversation's metadata key
 */

import { test, expect } from "@playwright/test";
import {
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  ensureMockLLMProfile,
  resetMockLLM,
  deleteConversation,
} from "../utils/mock-llm-helpers";
import * as fs from "fs";
import {
  getFolderBrowserPathSegments,
  getFolderBrowserRootPath,
  resolveFolderWorkspacePaths,
  TEST_DIR_NAME,
} from "../utils/folder-workspace-paths";

/**
 * The folder-workspace test creates a directory that the agent-server's folder
 * browser must be able to list.
 *
 * **Docker mode**: The Docker config volume-mounts a host dir into the
 * container at /tmp/e2e-folder-workspace-test, and sets two env vars:
 *   - MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR — host-side path for fs.mkdirSync
 *   - MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR — container-side path the
 *     folder browser navigates to
 *
 * **npm mode**: Host IS the agent-server, so both paths resolve identically
 * via os.tmpdir().
 */
const {
  hostDirBase: HOST_DIR_BASE,
  hostDir: HOST_DIR,
  testDir: TEST_DIR,
} = resolveFolderWorkspacePaths();

const METADATA_STORAGE_KEY = "openhands-agent-server-conversation-metadata";

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM folder browser → workspace → conversation", () => {
  const conversationIds = new Set<string>();

  test.beforeAll(async ({ browser }) => {
    // Create the test directory hierarchy (host-side path for Docker compat)
    fs.mkdirSync(HOST_DIR, { recursive: true });

    // Ensure the mock LLM profile is configured so conversations can start.
    // beforeAll only has worker-scoped fixtures, so create a temporary page.
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await ensureMockLLMProfile(page);
    } finally {
      await page.close();
    }
  });

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    await resetMockLLM(request);

    // Cleanup conversations created during the test
    for (const id of conversationIds) {
      try {
        await deleteConversation(request, id);
      } catch {
        // best-effort
      }
    }
    conversationIds.clear();
  });

  test.afterAll(async () => {
    // Remove the test directory (host-side path)
    try {
      fs.rmSync(HOST_DIR_BASE, { recursive: true, force: true });
    } catch {
      // best-effort
    }
  });

  // ── Step 1: Browse to a folder and add it as a workspace ────────────

  test("step 1: browse to a folder, add it as a workspace, and launch a conversation with the correct working_dir", async ({
    page,
  }) => {
    // Set up passive listener for POST /api/conversations BEFORE navigation.
    // Uses page.on('request') (not page.route) to avoid conflicts with
    // routeSessionApiKey — only one handler can call continue() per request.
    let capturedPayload: Record<string, unknown> | null = null;
    const captureConversationPayload = (
      req: import("@playwright/test").Request,
    ) => {
      if (
        req.method() === "POST" &&
        new URL(req.url()).pathname === "/api/conversations"
      ) {
        try {
          capturedPayload = req.postDataJSON();
        } catch {
          // non-JSON body
        }
      }
    };
    page.on("request", captureConversationPayload);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    // ── Open the "Open Workspace" dialog ──
    await test.step("open workspace dialog", async () => {
      await page.getByTestId("open-workspace-button").click();
      await expect(page.getByTestId("open-workspace-dialog-body")).toBeVisible({
        timeout: 10_000,
      });
    });

    // ── Browse to the test directory using the folder browser UI ──
    await test.step("open folder browser and navigate to test directory", async () => {
      // The "Add Workspaces" button is inside the dropdown's sticky footer,
      // so we must open the dropdown first.
      await page.getByTestId("workspace-dropdown").click();
      await page.getByTestId("add-workspaces-button").click();
      await expect(page.getByTestId("folder-browser-modal")).toBeVisible({
        timeout: 10_000,
      });

      // Navigate up to root first — click the "up" button repeatedly
      // until we reach "/" (path shows "/" or up button is disabled).
      const upBtn = page.getByTestId("folder-browser-up");
      const currentPathEl = page.getByTestId("folder-browser-current-path");
      const rootPath = getFolderBrowserRootPath(TEST_DIR);

      // Wait for the modal to finish initializing. `currentPath` starts as
      // null (rendering an empty path and a disabled up button) until
      // useHomeDirectory resolves and seeds the home path via useEffect.
      // Without this wait the while-loop below can see the briefly-disabled
      // up button and exit immediately, leaving us stuck at home instead of
      // navigating to root.
      await expect(currentPathEl).not.toHaveText("", { timeout: 10_000 });

      // Keep clicking up until the button becomes disabled (at root).
      while (!(await upBtn.isDisabled())) {
        await upBtn.click();
        await page.waitForTimeout(300);
      }
      await expect(currentPathEl).toHaveText(rootPath, { timeout: 5_000 });

      // Navigate down through each segment of the test directory path.
      // e.g. /tmp/e2e-folder-workspace-test/my-test-project → ["tmp", "e2e-...", "my-test-project"]
      const segments = getFolderBrowserPathSegments(TEST_DIR);
      for (const segment of segments) {
        const entry = page.getByTestId(`folder-browser-entry-${segment}`);
        await expect(entry).toBeVisible({ timeout: 10_000 });
        await entry.click();
      }

      // Verify we're at the correct path
      await expect(currentPathEl).toHaveText(TEST_DIR, { timeout: 5_000 });

      // Click "Use this folder"
      await page.getByTestId("folder-browser-use").click();

      // Modal should close
      await expect(page.getByTestId("folder-browser-modal")).toBeHidden({
        timeout: 5_000,
      });
    });

    // ── Confirm the selected workspace ──
    // The workspace dialog is still open after the folder browser closed.
    // Adding the folder auto-selects the new workspace, so wait for that
    // selection instead of reopening the dropdown and looking for an option
    // that is only rendered while the menu is open.
    await test.step("confirm the auto-selected workspace", async () => {
      await expect(page.getByTestId("open-workspace-dialog-body")).toBeVisible({
        timeout: 10_000,
      });

      const dropdown = page.getByTestId("workspace-dropdown");
      await expect(dropdown).toBeVisible({ timeout: 10_000 });
      await expect(dropdown).toHaveValue(TEST_DIR_NAME, { timeout: 10_000 });

      const confirmBtn = page.getByRole("button", { name: /confirm/i });
      await confirmBtn.click();

      await expect(page.getByTestId("open-workspace-dialog-body")).toBeHidden({
        timeout: 5_000,
      });
    });

    // ── Type a message and submit to create a conversation ──
    await test.step("submit a message to create a conversation", async () => {
      // Type into the home-page chat input (contentEditable div)
      const chatInput = page
        .getByTestId("home-chat-launcher")
        .locator('[contenteditable="true"]');
      await expect(chatInput).toBeVisible({ timeout: 10_000 });
      await chatInput.click();

      await page.evaluate((msg: string) => {
        const el = document.querySelector(
          '[data-testid="home-chat-launcher"] [contenteditable="true"]',
        );
        if (el) {
          el.textContent = msg;
          el.dispatchEvent(new Event("input", { bubbles: true }));
        }
      }, "Hello from the workspace test");

      // Submit with Enter
      await chatInput.press("Enter");

      // Wait for navigation to a conversation page
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    // Track the conversation for cleanup
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    const conversationId = match?.[1] ? decodeURIComponent(match[1]) : null;
    expect(conversationId, "Should be on a conversation page").toBeTruthy();
    conversationIds.add(conversationId!);

    // ── Verify: POST /api/conversations payload has correct working_dir ──
    await test.step("verify working_dir in POST /api/conversations payload", async () => {
      expect(
        capturedPayload,
        "POST /api/conversations payload was not captured",
      ).not.toBeNull();

      const workspace = capturedPayload?.workspace as
        | Record<string, unknown>
        | undefined;
      expect(workspace, "payload should have a workspace object").toBeTruthy();
      expect(workspace?.working_dir).toBe(TEST_DIR);
    });

    // ── Verify: selected_workspace in localStorage ──
    await test.step("verify selected_workspace in localStorage", async () => {
      const metadata = await page.evaluate(
        ({ key, convId }) => {
          const raw = window.localStorage.getItem(key);
          if (!raw) return null;
          try {
            const parsed = JSON.parse(raw);
            return parsed[convId] ?? null;
          } catch {
            return null;
          }
        },
        { key: METADATA_STORAGE_KEY, convId: conversationId! },
      );

      expect(
        metadata,
        `localStorage metadata for conversation ${conversationId} should exist`,
      ).not.toBeNull();
      expect(metadata?.selected_workspace).toBe(TEST_DIR);
    });

    page.off("request", captureConversationPayload);
  });
});
