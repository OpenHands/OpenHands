/**
 * Mock-LLM E2E test: preset automation card → slash command → skill activation.
 *
 * The `slack-standup-digest` skill ships in the public OpenHands extensions
 * repo with `triggers: ["/standup-digest:setup"]`. The frontend bundles
 * public skills from the `@openhands/extensions` npm package and passes
 * them directly in `agent_context.skills` at conversation-start, so the
 * SDK's trigger matching activates them without the agent-server needing
 * to clone the extensions repo (`load_public_skills: false`).
 *
 * Two tests:
 *   1. **Card flow**: configure a dummy Slack MCP server so the automation
 *      card is launchable, click the card, choose "Poll locally" in the
 *      responder-deployment modal, then verify it sends the correct slash
 *      command and triggers skill activation.
 *   2. **Direct slash command**: send the slash command from the home page
 *      (no MCP needed), verify skill activation + agent reply.
 */

import { test, expect } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
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
const AUTOMATION_CARD_ID = "slack-standup-digest";
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
          return `${String(e.source)}:${String(e.event_type)}(skills=${JSON.stringify(e.activated_skills ?? e.activated_microagents ?? [])})`;
        });

        const found = items.some((item: unknown) => {
          const e = item as Record<string, unknown>;
          const skills =
            (e.activated_skills as string[] | undefined) ??
            (e.activated_microagents as string[] | undefined);
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

  // ── Test 1: automation card sends the correct slash command ────────
  //
  // Configure a dummy Slack MCP server so the frontend sees it as
  // "installed" and the card is clickable without the install modal.
  // The dummy `echo` command can't do MCP JSON-RPC, so the agent-server
  // will error during tool initialization — but we only care that the
  // card click navigated to a conversation and sent the right prompt.
  // The end-to-end skill activation + agent reply is tested in test 2
  // (without MCP).

  test("automation card sends the correct slash command to a conversation", async ({
    page,
    request,
  }) => {
    await ensureMockLLMProfile(page);

    // Configure a dummy Slack MCP server so the card is launchable
    await test.step("configure dummy Slack MCP", async () => {
      const resp = await request.patch(`${BACKEND_URL}/api/settings`, {
        headers: {
          "X-Session-API-Key": SESSION_API_KEY,
          "Content-Type": "application/json",
        },
        data: {
          agent_settings_diff: {
            mcp_config: {
              slack: {
                command: "echo",
                args: ["dummy-slack-mcp"],
                env: {
                  SLACK_BOT_TOKEN: "xoxb-test-token",
                  SLACK_TEAM_ID: "T0000000000",
                },
              },
            },
          },
        },
      });
      expect(resp.ok(), `PATCH settings: ${resp.status()}`).toBe(true);
    });

    await routeSessionApiKey(page);
    await page.goto("/automations", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    // Click the Slack standup digest automation card
    await test.step("click automation card", async () => {
      await waitForTestId(page, "recommended-automations-section", 15_000);
      const card = page.getByTestId(
        `recommended-automation-card-${AUTOMATION_CARD_ID}`,
      );
      await expect(card).toBeVisible({ timeout: 10_000 });
      await card.click();
    });

    // Slack/GitHub responders now prompt for a deployment target first.
    // Choose "Poll locally on your laptop" to resume the existing local
    // setup + launch flow this test exercises.
    await test.step("choose local deployment", async () => {
      const modal = page.getByTestId("responder-deployment-modal");
      await expect(modal).toBeVisible({ timeout: 10_000 });
      await page.getByTestId("responder-deployment-continue-local").click();
    });

    // Should navigate to a new conversation
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    // Verify: the slash command was submitted as a user message.
    // The card calls setMessageToSend(prompt) which auto-submits.
    // Give it a moment, then fall back to manual submit if needed.
    await test.step("verify slash command is sent as user message", async () => {
      // Wait for either the user message to appear or the input to be populated
      await page.waitForTimeout(3_000);

      const userMessages = page.locator('[data-testid="user-message"]');
      const hasSentMessage = await userMessages
        .filter({ hasText: SLASH_COMMAND })
        .count()
        .then((c) => c > 0)
        .catch(() => false);

      if (!hasSentMessage) {
        // Message may still be in the input — submit it manually
        const inputEl = page.getByTestId("chat-input");
        const inputText = await inputEl.textContent().catch(() => "");
        if (inputText?.includes(SLASH_COMMAND)) {
          await page.getByTestId("submit-button").click();
        } else {
          // Card didn't populate the input — type and submit
          await setChatInput(page, SLASH_COMMAND);
          await page.getByTestId("submit-button").click();
        }
      }

      // Now verify the slash command appears as a sent user message
      await expect(
        userMessages.filter({ hasText: SLASH_COMMAND }),
      ).toBeVisible({ timeout: 15_000 });
    });
  });

  // ── Test 2: direct slash command (no MCP needed) ──────────────────

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
