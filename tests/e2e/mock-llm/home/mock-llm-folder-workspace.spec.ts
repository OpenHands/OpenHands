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

import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { test, expect, type APIRequestContext } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  setChatInput,
  ensureMockLLMProfile,
  resetMockLLM,
  deleteConversation,
} from "../utils/mock-llm-helpers";
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
 * under a unique, sentinel-owned child of the worktree root. The folder
 * browser hides dot-directories, so this path must remain visible.
 */
const RUN_DIR_NAME = `issue-16714-${process.pid}-${randomUUID()}`;
const WORKSPACE_PATH_ENV =
  process.env.MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR ||
  process.env.MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR
    ? process.env
    : {
        ...process.env,
        MOCK_LLM_FOLDER_WORKSPACE_HOST_DIR: resolve("."),
        MOCK_LLM_FOLDER_WORKSPACE_CONTAINER_DIR: resolve("."),
      };
const {
  hostDirBase: HOST_DIR_BASE,
  hostRunDir: HOST_RUN_DIR,
  hostDir: HOST_DIR,
  testDir: TEST_DIR,
} = resolveFolderWorkspacePaths({
  env: WORKSPACE_PATH_ENV,
  runDirName: RUN_DIR_NAME,
});

const METADATA_STORAGE_KEY = "openhands-agent-server-conversation-metadata";
const RUN_SENTINEL_NAME = ".openhands-e2e-owner";
const RUN_SENTINEL_CONTENT = `${JSON.stringify({
  task: "issue-16714",
  runDirName: RUN_DIR_NAME,
})}\n`;

function assertDirectChild(baseDir: string, candidateDir: string): void {
  const relativePath = relative(baseDir, candidateDir);
  if (
    !relativePath ||
    relativePath === ".." ||
    relativePath.startsWith(`..${sep}`) ||
    isAbsolute(relativePath) ||
    relativePath.includes(sep)
  ) {
    throw new Error(
      `Refusing to remove non-owned E2E path outside its direct base child: ${candidateDir}`,
    );
  }
}

function createOwnedRunDirectory(): void {
  fs.mkdirSync(HOST_DIR_BASE, { recursive: true });
  assertDirectChild(HOST_DIR_BASE, HOST_RUN_DIR);
  fs.mkdirSync(HOST_RUN_DIR);
  fs.writeFileSync(
    join(HOST_RUN_DIR, RUN_SENTINEL_NAME),
    RUN_SENTINEL_CONTENT,
    { encoding: "utf8", flag: "wx", mode: 0o600 },
  );
}

function removeOwnedRunDirectory(): void {
  if (!fs.existsSync(HOST_RUN_DIR)) return;

  const runStat = fs.lstatSync(HOST_RUN_DIR);
  if (!runStat.isDirectory() || runStat.isSymbolicLink()) {
    throw new Error(
      `Refusing to remove non-directory E2E path: ${HOST_RUN_DIR}`,
    );
  }

  const realBase = fs.realpathSync(HOST_DIR_BASE);
  const realRunDir = fs.realpathSync(HOST_RUN_DIR);
  assertDirectChild(realBase, realRunDir);

  const sentinelPath = join(realRunDir, RUN_SENTINEL_NAME);
  const sentinelStat = fs.lstatSync(sentinelPath);
  if (!sentinelStat.isFile() || sentinelStat.isSymbolicLink()) {
    throw new Error(`Refusing to remove E2E path without its task sentinel`);
  }
  if (fs.readFileSync(sentinelPath, "utf8") !== RUN_SENTINEL_CONTENT) {
    throw new Error(`Refusing to remove E2E path owned by another run`);
  }

  fs.rmSync(realRunDir, { recursive: true, force: true });
}

function isPathAtOrBelow(path: string, parent: string): boolean {
  const normalize = (value: string) =>
    value.replaceAll("\\", "/").replace(/\/+$/, "");
  const normalizedPath = normalize(path);
  const normalizedParent = normalize(parent);
  const caseInsensitive = /^[A-Za-z]:\//.test(normalizedPath);
  const target = caseInsensitive
    ? normalizedPath.toLowerCase()
    : normalizedPath;
  const base = caseInsensitive
    ? normalizedParent.toLowerCase()
    : normalizedParent;
  return target === base || target.startsWith(`${base}/`);
}

async function setWorktreeDefault(
  request: APIRequestContext,
  enabled: boolean,
) {
  const response = await request.patch(`${BACKEND_URL}/api/settings`, {
    headers: {
      "X-Session-API-Key": SESSION_API_KEY,
      "Content-Type": "application/json",
    },
    data: {
      misc_settings_diff: {
        app_preferences: { use_worktree_by_default: enabled },
      },
    },
  });
  expect(
    response.ok(),
    `Failed to set worktree preference: ${response.status()}`,
  ).toBe(true);
}

async function getWorktreeDefault(
  request: APIRequestContext,
): Promise<boolean> {
  const response = await request.get(`${BACKEND_URL}/api/settings`, {
    headers: { "X-Session-API-Key": SESSION_API_KEY },
  });
  expect(
    response.ok(),
    `Failed to read worktree preference: ${response.status()}`,
  ).toBe(true);
  const body = (await response.json()) as {
    misc_settings?: {
      app_preferences?: { use_worktree_by_default?: unknown };
    };
  };
  return body.misc_settings?.app_preferences?.use_worktree_by_default === true;
}

async function removeWorkspace(
  request: APIRequestContext,
  path: string,
): Promise<void> {
  const response = await request.delete(`${BACKEND_URL}/api/workspaces`, {
    headers: { "X-Session-API-Key": SESSION_API_KEY },
    params: { path },
  });
  if (!response.ok() && response.status() !== 404) {
    throw new Error(`Failed to remove test workspace: ${response.status()}`);
  }
}

async function registerWorkspace(request: APIRequestContext): Promise<void> {
  const response = await request.post(`${BACKEND_URL}/api/workspaces`, {
    headers: {
      "X-Session-API-Key": SESSION_API_KEY,
      "Content-Type": "application/json",
    },
    data: {
      workspaces: [
        { id: `e2e-${RUN_DIR_NAME}`, name: TEST_DIR_NAME, path: TEST_DIR },
      ],
    },
  });
  expect(
    response.ok(),
    `Failed to register workspace: ${response.status()} ${await response.text()}`,
  ).toBe(true);
}

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM folder browser → workspace → conversation", () => {
  const conversationIds = new Set<string>();
  let originalWorktreeDefault: boolean | null = null;

  const deleteTrackedConversations = async (request: APIRequestContext) => {
    const errors: unknown[] = [];
    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch (error) {
        errors.push(error);
      }
    }
    return errors;
  };

  test.beforeAll(async ({ browser, request }) => {
    originalWorktreeDefault = await getWorktreeDefault(request);

    // Create a committed Git workspace so the agent-server can cut real
    // worktrees from it. HOST_DIR is the host-side path in Docker mode.
    createOwnedRunDirectory();
    fs.mkdirSync(HOST_DIR);
    fs.writeFileSync(join(HOST_DIR, "README.md"), "# Worktree E2E\n");
    execFileSync("git", ["init", "--initial-branch=main", HOST_DIR]);
    execFileSync("git", [
      "-C",
      HOST_DIR,
      "config",
      "user.name",
      "OpenHands E2E",
    ]);
    execFileSync("git", [
      "-C",
      HOST_DIR,
      "config",
      "user.email",
      "e2e@localhost",
    ]);
    execFileSync("git", ["-C", HOST_DIR, "add", "README.md"]);
    execFileSync("git", ["-C", HOST_DIR, "commit", "-m", "Initial commit"]);

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

  test.beforeEach(async ({ page, request }) => {
    await setWorktreeDefault(request, false);
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ request }) => {
    const cleanupErrors = await deleteTrackedConversations(request);

    // Do not reset the shared FIFO while an owned conversation may still be
    // running. afterAll retries any failed deletions before removing its repo.
    if (conversationIds.size === 0) {
      try {
        await resetMockLLM(request);
      } catch (error) {
        cleanupErrors.push(error);
      }
    }

    try {
      await setWorktreeDefault(request, false);
    } catch (error) {
      cleanupErrors.push(error);
    }

    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        cleanupErrors,
        "Failed to clean up the folder-workspace E2E test",
      );
    }
  });

  test.afterAll(async ({ request }) => {
    const cleanupErrors = await deleteTrackedConversations(request);

    if (conversationIds.size === 0) {
      try {
        await removeWorkspace(request, TEST_DIR);
        // Never delete the bind-mount root: only this sentinel-owned run child.
        removeOwnedRunDirectory();
      } catch (error) {
        cleanupErrors.push(error);
      }
    } else {
      cleanupErrors.push(
        new Error(
          "Preserving the test workspace because an owned conversation is still running",
        ),
      );
    }

    try {
      if (originalWorktreeDefault !== null) {
        await setWorktreeDefault(request, originalWorktreeDefault);
      }
    } catch (error) {
      cleanupErrors.push(error);
    }

    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        cleanupErrors,
        "Failed to tear down the folder-workspace E2E suite",
      );
    }
  });

  // ── Step 1: Browse to a folder and add it as a workspace ────────────

  test("step 1: browse to a folder, add it as a workspace, and launch a conversation with the correct working_dir", async ({
    page,
  }) => {
    test.setTimeout(120_000);
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

      const initialPath = (await currentPathEl.textContent()) ?? "";
      let startingSegmentCount = 0;

      if (isPathAtOrBelow(TEST_DIR, initialPath)) {
        startingSegmentCount = getFolderBrowserPathSegments(initialPath).length;
      } else {
        // Keep clicking up until the button becomes disabled (at root).
        while (!(await upBtn.isDisabled())) {
          await upBtn.click();
          await page.waitForTimeout(300);
        }
        await expect(currentPathEl).toHaveText(rootPath, { timeout: 5_000 });
      }

      // Navigate down through each segment of the test directory path.
      // e.g. /tmp/e2e-folder-workspace-test/my-test-project → ["tmp", "e2e-...", "my-test-project"]
      const segments =
        getFolderBrowserPathSegments(TEST_DIR).slice(startingSegmentCount);
      for (const segment of segments) {
        const entry = page.getByTestId(`folder-browser-entry-${segment}`);
        await expect(entry).toBeVisible({ timeout: 30_000 });
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
    let conversationId: string | null = null;
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

      const createResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/conversations",
      );

      // Submit with Enter
      await chatInput.press("Enter");

      const createResponse = await createResponsePromise;
      expect(createResponse.ok()).toBe(true);
      const createBody = (await createResponse.json()) as { id?: unknown };
      expect(typeof createBody.id).toBe("string");
      conversationId = createBody.id as string;
      conversationIds.add(conversationId);

      // Wait for navigation to a conversation page
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    // Verify navigation used the same conversation that was already registered
    // for cleanup from the successful create response.
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    const navigatedConversationId = match?.[1]
      ? decodeURIComponent(match[1])
      : null;
    expect(navigatedConversationId).toBe(conversationId);

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
      expect(capturedPayload?.worktree).toBe(false);
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

  test("saved worktree default creates an isolated workspace and still allows a local-repo override", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);
    await removeWorkspace(request, TEST_DIR);
    await registerWorkspace(request);

    const capturedPayloads: Record<string, unknown>[] = [];
    const captureConversationPayload = (
      req: import("@playwright/test").Request,
    ) => {
      if (
        req.method() === "POST" &&
        new URL(req.url()).pathname === "/api/conversations"
      ) {
        capturedPayloads.push(req.postDataJSON() as Record<string, unknown>);
      }
    };
    page.on("request", captureConversationPayload);

    await routeSessionApiKey(page);
    await page.goto("/settings/app", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "app-settings-screen");

    await test.step("save the default-worktree preference through Settings", async () => {
      const worktreeSwitch = page.getByTestId("use-worktree-by-default-switch");
      await expect(worktreeSwitch).not.toBeChecked();
      await worktreeSwitch.locator("xpath=..").click();
      await expect(worktreeSwitch).toBeChecked();

      const patchResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "PATCH" &&
          new URL(response.url()).pathname === "/api/settings",
      );
      await page.getByTestId("submit-button").click();
      const patchResponse = await patchResponsePromise;
      expect(patchResponse.ok()).toBe(true);
      expect(patchResponse.request().postDataJSON()).toMatchObject({
        misc_settings_diff: {
          app_preferences: { use_worktree_by_default: true },
        },
      });

      await expect
        .poll(async () => {
          const response = await request.get(`${BACKEND_URL}/api/settings`, {
            headers: { "X-Session-API-Key": SESSION_API_KEY },
          });
          if (!response.ok()) return null;
          const body = await response.json();
          return body.misc_settings?.app_preferences?.use_worktree_by_default;
        })
        .toBe(true);
    });

    const selectWorkspace = async () => {
      await page.getByTestId("open-workspace-button").click();
      const dropdown = page.getByTestId("workspace-dropdown");
      await dropdown.click();
      await page
        .getByTestId("workspace-dropdown-menu")
        .getByText(TEST_DIR_NAME, { exact: true })
        .click();
      await page.getByTestId("workspace-launch-button").click();
      await expect(page.getByTestId("open-workspace-dialog-body")).toBeHidden();
    };

    const getRuntimeWorkingDir = async (conversationId: string) => {
      const response = await request.get(
        `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}`,
        { headers: { "X-Session-API-Key": SESSION_API_KEY } },
      );
      expect(
        response.ok(),
        `GET conversation failed: ${response.status()}`,
      ).toBe(true);
      const body = await response.json();
      expect(body.workspace?.kind).toBe("LocalWorkspace");
      return body.workspace?.working_dir as string | undefined;
    };

    const goHomeWithFreshSettings = async () => {
      const settingsResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "GET" &&
          new URL(response.url()).pathname === "/api/settings" &&
          response.ok(),
      );
      await page.goto("/", { waitUntil: "domcontentloaded" });
      await settingsResponsePromise;
      await waitForTestId(page, "home-chat-launcher");
    };

    await test.step("launch with the saved worktree default", async () => {
      // Full navigation starts with a fresh query cache, proving the preference
      // was persisted by the server rather than retained only in component state.
      await goHomeWithFreshSettings();
      await selectWorkspace();

      const modeSelector = page.getByTestId("workspace-mode-selector");
      await modeSelector.click();
      await expect(
        page.getByTestId("workspace-mode-selector-option-new_worktree"),
      ).toHaveAttribute("aria-checked", "true");
      await page.keyboard.press("Escape");

      await setChatInput(page, "Start in the default worktree");
      const createResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/conversations",
      );
      await page.getByTestId("submit-button").click();
      const createResponse = await createResponsePromise;
      expect(createResponse.ok()).toBe(true);
      const createBody = (await createResponse.json()) as { id?: unknown };
      expect(typeof createBody.id).toBe("string");
      const conversationId = createBody.id as string;
      conversationIds.add(conversationId);
      await waitForPath(page, /\/conversations\/.+/, 30_000);

      expect(getConversationIdFromURL(page)).toBe(conversationId);
      expect(capturedPayloads[0]?.worktree).toBe(true);
      const requestWorkspace = capturedPayloads[0]?.workspace as
        | Record<string, unknown>
        | undefined;
      expect(requestWorkspace?.working_dir).toBe(TEST_DIR);

      const runtimeWorkingDir = await getRuntimeWorkingDir(conversationId);
      expect(runtimeWorkingDir).toBeTruthy();
      expect(runtimeWorkingDir).not.toBe(TEST_DIR);

      // Stop the first agent before resetting the shared mock trajectory for
      // the second launch. This keeps the two runtime-workspace assertions
      // independent without changing the mock server's global FIFO contract.
      await deleteConversation(request, conversationId);
      conversationIds.delete(conversationId);
      await resetMockLLM(request);
    });

    await test.step("override the preference with Local Repo for one conversation", async () => {
      await goHomeWithFreshSettings();
      await selectWorkspace();

      const modeSelector = page.getByTestId("workspace-mode-selector");
      await modeSelector.click();
      await page
        .getByTestId("workspace-mode-selector-option-local_repo")
        .click();

      await setChatInput(page, "Use the source repository this time");
      const createResponsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/conversations",
      );
      await page.getByTestId("submit-button").click();
      const createResponse = await createResponsePromise;
      expect(createResponse.ok()).toBe(true);
      const createBody = (await createResponse.json()) as { id?: unknown };
      expect(typeof createBody.id).toBe("string");
      const conversationId = createBody.id as string;
      conversationIds.add(conversationId);
      await waitForPath(page, /\/conversations\/.+/, 30_000);

      expect(getConversationIdFromURL(page)).toBe(conversationId);
      expect(capturedPayloads[1]?.worktree).toBe(false);
      const runtimeWorkingDir = await getRuntimeWorkingDir(conversationId);
      expect(runtimeWorkingDir).toBe(TEST_DIR);
    });

    page.off("request", captureConversationPayload);
  });
});
