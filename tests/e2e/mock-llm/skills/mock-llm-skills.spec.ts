/**
 * Mock-LLM E2E tests: skill loading from project and user directories.
 *
 * These tests verify that the SDK's skill loading machinery works
 * end-to-end with the real agent-server stack:
 *
 * 1. **Project skills** from `{workspace}/.agents/skills/` are loaded
 *    alongside bundled public skills and trigger on matching keywords.
 *    The test creates a standalone git repo with the skill committed,
 *    then creates a conversation via API pointing at that repo. The
 *    agent-server creates a worktree from the repo, and since the skill
 *    is committed, `load_project_skills` finds it in the worktree.
 *
 * 2. **User skills** from `~/.openhands/skills/` are loaded and trigger
 *    on matching keywords.
 *
 * 3. **Skill deletion**: removing a skill file means it is NOT loaded
 *    in subsequent conversations.
 *
 * All tests create ephemeral SKILL.md files with unique trigger keywords,
 * send a message containing those keywords, and verify `activated_skills`
 * appears in the conversation events API.
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
  waitForNonUserMessageText,
  deleteConversation,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  ensureMockLLMProfile,
  setChatInput,
  waitForPath,
  getConversationIdFromURL,
} from "../utils/mock-llm-helpers";
import {
  createProjectSkillRepo,
  removeProjectSkillRepo,
  writeUserSkill,
  removeUserSkill,
  userSkillExists,
  userSkillDirExists,
} from "../utils/skill-test-helpers";

/**
 * Register a workspace on the agent-server so it appears in the UI dropdown.
 */
async function addWorkspaceToServer(
  request: APIRequestContext,
  name: string,
  path: string,
): Promise<void> {
  const resp = await request.post(`${BACKEND_URL}/api/workspaces`, {
    headers: {
      "X-Session-API-Key": SESSION_API_KEY,
      "Content-Type": "application/json",
    },
    data: {
      workspaces: [{ id: `e2e-${name}`, name, path }],
    },
  });
  expect(
    resp.ok(),
    `POST /api/workspaces failed: ${resp.status()} ${await resp.text()}`,
  ).toBe(true);
}

/**
 * Remove a workspace from the agent-server.
 */
async function removeWorkspaceFromServer(
  request: APIRequestContext,
  path: string,
): Promise<void> {
  await request.delete(`${BACKEND_URL}/api/workspaces`, {
    headers: { "X-Session-API-Key": SESSION_API_KEY },
    params: { path },
  });
}

async function navigateToNewChat(page: Page): Promise<void> {
  await expect(async () => {
    await page
      .getByTestId("sidebar-conversations-link")
      .click({ timeout: 5_000 });
    await expect(page).toHaveURL(/\/conversations\/?$/, { timeout: 5_000 });
  }).toPass({ timeout: 30_000, intervals: [500, 1_000, 2_000] });
}

// ── Shared constants ─────────────────────────────────────────────────

const REPLY_TOKEN = "SKILLS_E2E_REPLY_OK";

// ── Tests ─────────────────────────────────────────────────────────────

test.describe.configure({ mode: "serial" });

test.describe("skill loading: project, user, and deletion", () => {
  const conversationIds = new Set<string>();

  // Unique skill names to avoid collisions with real skills
  const PROJECT_SKILL_NAME = "e2e-test-project-skill";
  const PROJECT_SKILL_TRIGGER = "xyzzy-project-e2e-test";

  const USER_SKILL_NAME = "e2e-test-user-skill";
  const USER_SKILL_TRIGGER = "xyzzy-user-e2e-test";

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ page, request }) => {
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    if (match?.[1]) conversationIds.add(decodeURIComponent(match[1]));

    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch {
        // best-effort cleanup
      }
    }
    await resetMockLLM(request).catch(() => {});
  });

  // Track the agent-side workspace path for cleanup (may differ from
  // the host path in Docker mode where volumes are mounted)
  let projectSkillAgentDir = "";

  test.afterAll(async ({ request }) => {
    removeProjectSkillRepo(PROJECT_SKILL_NAME);
    removeUserSkill(USER_SKILL_NAME);
    if (projectSkillAgentDir) {
      await removeWorkspaceFromServer(request, projectSkillAgentDir).catch(
        () => {},
      );
    }
  });

  // ── Test 1: Project skill loaded from workspace ──────────────────
  //
  // Creates a standalone git repo with the skill committed, registers it
  // as a workspace on the agent-server, then uses the UI to select that
  // workspace and send a message. The agent-server creates a worktree
  // from the repo, and the committed skill file is present in it for
  // `load_project_skills` to discover.

  test("project skill in workspace/.agents/skills/ triggers on matching keyword", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    // Create a git repo with the skill committed
    const { agentDir } =
      await test.step("create git repo with project skill", () => {
        return createProjectSkillRepo(
          PROJECT_SKILL_NAME,
          PROJECT_SKILL_TRIGGER,
        );
      });
    projectSkillAgentDir = agentDir;

    // Register the workspace on the server using the agent-side path
    // (same as hostDir in npm mode, container mount path in Docker mode)
    await test.step("register workspace on server", async () => {
      await addWorkspaceToServer(request, "skill-test-repo", agentDir);
    });

    // Trajectory: padding for skill-analysis + agent reply
    await registerTrajectory(request, "project-skill", [
      { text: "" },
      { text: `Skill test complete. ${REPLY_TOKEN}` },
    ]);
    await activateTrajectory(request, "project-skill");

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Select the workspace through the UI
    await test.step("select workspace from UI", async () => {
      // Click "Open workspace" to open the dialog
      await page.getByTestId("open-workspace-button").click();

      // Click the workspace dropdown and select our workspace
      const dropdown = page.getByTestId("workspace-dropdown");
      await dropdown.click();
      // The workspace name is "skill-test-repo" — click the matching option
      await page.getByText("skill-test-repo").click();

      // Click the Confirm button to set the workspace
      await page.getByTestId("workspace-launch-button").click();
    });

    await test.step("send message with project skill trigger", async () => {
      await setChatInput(
        page,
        `Please help me with ${PROJECT_SKILL_TRIGGER} setup`,
      );
      await page.getByTestId("submit-button").click();
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    await test.step("verify agent reply", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 45_000);
    });

    await test.step("verify project skill activated", async () => {
      await expect
        .poll(
          async () => {
            const resp = await request.get(
              `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
              {
                headers: { "X-Session-API-Key": SESSION_API_KEY },
                params: { limit: "50" },
              },
            );
            if (!resp.ok()) return `HTTP ${resp.status()}`;
            const body = (await resp.json()) as { items?: unknown[] };
            for (const item of body.items ?? []) {
              const e = item as Record<string, unknown>;
              const skills =
                (e.activated_skills as string[] | undefined) ??
                (e.activated_microagents as string[] | undefined);
              if (skills?.includes(PROJECT_SKILL_NAME)) return "FOUND";
            }
            return "NOT_FOUND";
          },
          {
            message: `expected "${PROJECT_SKILL_NAME}" in activated_skills`,
            intervals: [1_000, 2_000, 3_000, 5_000],
            timeout: 25_000,
          },
        )
        .toBe("FOUND");
    });
  });

  // ── Test 2: User skill loaded from ~/.openhands/skills/ ──────────

  test("user skill in ~/.openhands/skills/ triggers on matching keyword", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    await test.step("create user skill file", () => {
      writeUserSkill(USER_SKILL_NAME, USER_SKILL_TRIGGER);
      expect(userSkillExists(USER_SKILL_NAME)).toBe(true);
    });

    await registerTrajectory(request, "user-skill", [
      { text: "" },
      { text: `Skill test complete. ${REPLY_TOKEN}` },
    ]);
    await activateTrajectory(request, "user-skill");

    await routeSessionApiKey(page);
    await navigateToNewChat(page);

    await test.step("send message with user skill trigger", async () => {
      await setChatInput(
        page,
        `I need help with ${USER_SKILL_TRIGGER} configuration`,
      );
      await page.getByTestId("submit-button").click();
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    await test.step("verify agent reply", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 45_000);
    });

    await test.step("verify user skill activated", async () => {
      await expect
        .poll(
          async () => {
            const resp = await request.get(
              `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
              {
                headers: { "X-Session-API-Key": SESSION_API_KEY },
                params: { limit: "50" },
              },
            );
            if (!resp.ok()) return `HTTP ${resp.status()}`;
            const body = (await resp.json()) as { items?: unknown[] };
            for (const item of body.items ?? []) {
              const e = item as Record<string, unknown>;
              const skills =
                (e.activated_skills as string[] | undefined) ??
                (e.activated_microagents as string[] | undefined);
              if (skills?.includes(USER_SKILL_NAME)) return "FOUND";
            }
            return "NOT_FOUND";
          },
          {
            message: `expected "${USER_SKILL_NAME}" in activated_skills`,
            intervals: [1_000, 2_000, 3_000, 5_000],
            timeout: 25_000,
          },
        )
        .toBe("FOUND");
    });
  });

  // ── Test 3: Deleted user skill not loaded in new conversation ────

  test("deleting a user skill removes it from subsequent conversations", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    await test.step("delete user skill file", () => {
      removeUserSkill(USER_SKILL_NAME);
      expect(userSkillDirExists(USER_SKILL_NAME)).toBe(false);
    });

    // Padding for skill-analysis (public skills are still loaded from the npm
    // package, so the agent-server still makes a skill-analysis LLM call)
    await registerTrajectory(request, "deleted-skill", [
      { text: "" },
      { text: `No skill triggered. ${REPLY_TOKEN}` },
    ]);
    await activateTrajectory(request, "deleted-skill");

    await routeSessionApiKey(page);
    await navigateToNewChat(page);

    await test.step("send message with deleted skill trigger keyword", async () => {
      await setChatInput(page, `Help me with ${USER_SKILL_TRIGGER} please`);
      await page.getByTestId("submit-button").click();
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    await test.step("verify agent reply", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 45_000);
    });

    await test.step("verify deleted skill NOT activated", async () => {
      // The agent reply already appeared in the UI (verified above), so the
      // conversation completed. Poll the events API and verify no event
      // contains the deleted skill in its activated_skills list.
      await expect
        .poll(
          async () => {
            const resp = await request.get(
              `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}/events/search`,
              {
                headers: { "X-Session-API-Key": SESSION_API_KEY },
                params: { limit: "100" },
              },
            );
            if (!resp.ok()) return `HTTP ${resp.status()}`;
            const body = (await resp.json()) as { items?: unknown[] };
            const items = body.items ?? [];
            if (items.length === 0) return "NO_EVENTS";

            for (const item of items) {
              const e = item as Record<string, unknown>;
              const skills =
                (e.activated_skills as string[] | undefined) ??
                (e.activated_microagents as string[] | undefined);
              if (skills?.includes(USER_SKILL_NAME))
                return `UNEXPECTEDLY_FOUND`;
            }
            return "VERIFIED";
          },
          {
            message: `verifying "${USER_SKILL_NAME}" NOT in activated_skills`,
            intervals: [1_000, 2_000, 3_000],
            timeout: 15_000,
          },
        )
        .toBe("VERIFIED");
    });
  });
});
