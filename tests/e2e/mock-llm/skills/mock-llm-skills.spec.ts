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
 * 2. **User skills** from the stack's persistence directory are loaded and trigger
 *    on matching keywords.
 *
 * 3. **Skill deletion**: removing a skill file means it is NOT loaded
 *    in subsequent conversations.
 *
 * All tests create ephemeral SKILL.md files with unique trigger keywords,
 * send a message containing those keywords, and verify `activated_skills`
 * appears in the conversation events API.
 */

import { test } from "../utils/atomic-journey";
import { expect, type APIRequestContext, type Page } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForNonUserMessageText,
  registerTrajectory,
  activateTrajectory,
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
  const response = await request.delete(`${BACKEND_URL}/api/workspaces`, {
    headers: { "X-Session-API-Key": SESSION_API_KEY },
    params: { path },
  });
  expect(
    response.ok() || response.status() === 404,
    `Remove workspace: ${response.status()}`,
  ).toBe(true);
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

test("project skill in workspace/.agents/skills/ triggers on matching keyword", async ({
  page,
  request,
  journey,
}) => {
  test.setTimeout(180_000);
  const PROJECT_SKILL_NAME = journey.profileName + "-project-skill";
  const PROJECT_SKILL_TRIGGER = "xyzzy-project-e2e-test";
  journey.cleanup("project skill files", () =>
    removeProjectSkillRepo(PROJECT_SKILL_NAME),
  );
  await ensureMockLLMProfile(page, { profileName: journey.profileName });

  // Create a git repo with the skill committed
  const { agentDir } =
    await test.step("create git repo with project skill", () => {
      return createProjectSkillRepo(PROJECT_SKILL_NAME, PROJECT_SKILL_TRIGGER);
    });
  journey.cleanup("registered workspace", () =>
    removeWorkspaceFromServer(request, agentDir),
  );

  // Register the workspace on the server using the agent-side path
  // (same as hostDir in npm mode, container mount path in Docker mode)
  await test.step("register workspace on server", async () => {
    await addWorkspaceToServer(request, PROJECT_SKILL_NAME, agentDir);
  });

  // Title generation does not consume the scripted agent reply.
  await registerTrajectory(request, "project-skill", [
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
    // Select this test attempt's registered workspace.
    await page.getByText(PROJECT_SKILL_NAME, { exact: true }).click();

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
  journey.conversationIds.add(conversationId);

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
            const skills = e.activated_skills as string[] | undefined;
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

test("user skill in ~/.openhands/skills/ triggers on matching keyword", async ({
  page,
  request,
  journey,
}) => {
  test.setTimeout(180_000);
  const USER_SKILL_NAME = journey.profileName + "-user-skill";
  const USER_SKILL_TRIGGER = "xyzzy-user-e2e-test";
  journey.cleanup("user skill files", () => removeUserSkill(USER_SKILL_NAME));
  await ensureMockLLMProfile(page, { profileName: journey.profileName });

  await test.step("create user skill file", () => {
    writeUserSkill(USER_SKILL_NAME, USER_SKILL_TRIGGER);
    expect(userSkillExists(USER_SKILL_NAME)).toBe(true);
  });

  await registerTrajectory(request, "user-skill", [
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
  journey.conversationIds.add(conversationId);

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
            const skills = e.activated_skills as string[] | undefined;
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

test("deleting a user skill removes it from subsequent conversations", async ({
  page,
  request,
  journey,
}) => {
  test.setTimeout(180_000);
  const USER_SKILL_NAME = journey.profileName + "-user-skill";
  const USER_SKILL_TRIGGER = "xyzzy-user-e2e-test";
  journey.cleanup("user skill files", () => removeUserSkill(USER_SKILL_NAME));
  await ensureMockLLMProfile(page, { profileName: journey.profileName });

  await test.step("create and verify user skill before deletion", () => {
    writeUserSkill(USER_SKILL_NAME, USER_SKILL_TRIGGER);
    expect(userSkillExists(USER_SKILL_NAME)).toBe(true);
  });

  await test.step("delete user skill file", () => {
    removeUserSkill(USER_SKILL_NAME);
    expect(userSkillDirExists(USER_SKILL_NAME)).toBe(false);
  });

  await registerTrajectory(request, "deleted-skill", [
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
  journey.conversationIds.add(conversationId);

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
            const skills = e.activated_skills as string[] | undefined;
            if (skills?.includes(USER_SKILL_NAME)) return `UNEXPECTEDLY_FOUND`;
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
