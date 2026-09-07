/**
 * Mock-LLM E2E test: preset automation slash command → skill activation.
 *
 * The `slack-standup-digest` skill ships in the public OpenHands extensions
 * repo with `triggers: ["/standup-digest:setup"]`. The frontend bundles
 * public skills from the `@openhands/extensions` npm package and passes
 * them directly in `agent_context.skills` at conversation-start, so the
 * SDK's trigger matching activates them without the agent-server needing
 * to clone the extensions repo (`load_public_skills: false`).
 *
 * The Slack-based automation *cards* are excluded from the recommended UI
 * (Ruckus has no Slack workspace), so this spec sends the slash command
 * directly from the home page (no MCP needed) and verifies skill
 * activation + agent reply.
 */

import { test, expect } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  deleteConversation,
  registerTrajectory,
  activateTrajectory,
  resetMockLLM,
  ensureMockLLMProfile,
  setChatInput,
} from "../utils/mock-llm-helpers";

const SLASH_COMMAND = "/standup-digest:setup";
const REPLY_TOKEN = "PRESET_AUTOMATION_REPLY_OK";

// ── Shared helpers ────────────────────────────────────────────────────

/** Register + activate the mock LLM trajectory for a skill-triggered conversation. */
async function setupTrajectory(
  request: import("@playwright/test").APIRequestContext,
) {
  // Response 0: padding for internal skill-analysis call (consumed when
  // a skill trigger matches, before the agent loop starts).
  // Response 1: the actual agent reply.
  await registerTrajectory(request, "preset-automation", [
    { text: "" },
    { text: `I'll help you set up the standup digest. ${REPLY_TOKEN}` },
  ]);
  await activateTrajectory(request, "preset-automation");
}

/** Poll the events API until we find an event with a non-empty activated_skills. */
async function assertActivatedSkills(
  request: import("@playwright/test").APIRequestContext,
  conversationId: string,
) {
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
        if (!resp.ok()) return `events API: HTTP ${resp.status()}`;

        const body = (await resp.json()) as { items?: unknown[] };
        const items = body.items ?? [];

        const diag = items.map((item: unknown) => {
          const e = item as Record<string, unknown>;
          return `${String(e.source)}:${String(e.event_type)}(skills=${JSON.stringify(e.activated_skills ?? [])})`;
        });

        const found = items.some((item: unknown) => {
          const e = item as Record<string, unknown>;
          const skills = e.activated_skills as string[] | undefined;
          return Array.isArray(skills) && skills.length > 0;
        });

        return found
          ? "FOUND"
          : `${items.length} events: [${diag.join(", ")}]`;
      },
      {
        message: "activated_skills not found in conversation events",
        intervals: [1_000, 2_000, 3_000, 5_000],
        timeout: 20_000,
      },
    )
    .toBe("FOUND");
}

// ── Tests ─────────────────────────────────────────────────────────────

test.describe.configure({ mode: "serial" });

test.describe("preset automation → slash command conversation", () => {
  const conversationIds = new Set<string>();

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
    // Clear any MCP servers so subsequent tests start clean
    await request
      .patch(`${BACKEND_URL}/api/settings`, {
        headers: {
          "X-Session-API-Key": SESSION_API_KEY,
          "Content-Type": "application/json",
        },
        data: { agent_settings_diff: { mcp_config: null } },
      })
      .catch(() => {});
  });

  // ── Test: direct slash command (no MCP needed) ──────────────────

  test("direct slash command from home page triggers skill activation", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    // Explicitly clear the MCP servers left by test 1.
    // Setting mcp_config to null removes it entirely (an empty {} is treated
    // as a no-op partial merge).
    const clearResp = await request.patch(`${BACKEND_URL}/api/settings`, {
      headers: {
        "X-Session-API-Key": SESSION_API_KEY,
        "Content-Type": "application/json",
      },
      data: {
        agent_settings_diff: { mcp_config: null },
      },
    });
    expect(clearResp.ok(), `Clear MCP: ${clearResp.status()}`).toBe(true);

    // Verify MCP is actually gone
    const settingsResp = await request.get(`${BACKEND_URL}/api/settings`, {
      headers: { "X-Session-API-Key": SESSION_API_KEY },
    });
    const settings = await settingsResp.json();
    const servers = settings?.agent_settings?.mcp_config ?? {};
    expect(
      Object.keys(servers).length,
      `MCP servers should be empty, got: ${JSON.stringify(servers).slice(0, 200)}`,
    ).toBe(0);

    await setupTrajectory(request);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Type and submit the slash command from the home page
    await test.step("send slash command from home page", async () => {
      await setChatInput(page, SLASH_COMMAND);
      await page.getByTestId("submit-button").click();
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    await test.step("verify user message", async () => {
      const userMessages = page.locator('[data-testid="user-message"]');
      await expect(
        userMessages.filter({ hasText: SLASH_COMMAND }),
      ).toBeVisible({ timeout: 15_000 });
    });

    await test.step("verify agent reply", async () => {
      await waitForNonUserMessageText(page, REPLY_TOKEN, 45_000);
    });

    await test.step("verify activated_skills in events", async () => {
      await assertActivatedSkills(request, conversationId);
    });
  });
});
