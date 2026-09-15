/**
 * Mock-LLM E2E: Canvas-enabled built-in skills reach a Claude Code ACP session.
 *
 * Native ACP sourcing strips ``agent_context.skills`` on profile launch
 * (OpenHands#16905). Canvas projects enabled catalog instructions through
 * ``agent_launch_additions.system_message_suffix_append``, which the
 * agent-server appends after that strip. This spec tags the profile as
 * Claude Code while pointing ``acp_command`` at the mock ACP server so CI
 * does not need a real Claude binary — then asserts the *resolved* conversation
 * agent (not only the frontend POST payload).
 */

import { test, expect } from "@playwright/test";
import {
  ACP_REPLY_TOKEN,
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
  waitForPath,
  getConversationIdFromURL,
  waitForNonUserMessageText,
  deleteConversation,
  resetToOpenHandsAgentViaUI,
  resetMockLLM,
  ensureMockLLMProfile,
  ensureMockLLMAgentProfile,
  ensureMockClaudeAcpAgentProfile,
  patchSkillEnablement,
  setChatInput,
  BACKEND_URL,
  SESSION_API_KEY,
} from "../utils/mock-llm-helpers";

const AUTOMATION_HEADING = "# OpenHands Automations";
const USER_MESSAGE = "Hello ACP agent, please reply.";

function sessionHeaders() {
  return { "X-Session-API-Key": SESSION_API_KEY };
}

async function readResolvedConversation(
  request: import("@playwright/test").APIRequestContext,
  conversationId: string,
) {
  // include_skills=true: GET otherwise trims skills to [] on the wire, which
  // would make an "ACP stripped skills" assertion meaningless.
  const resp = await request.get(
    `${BACKEND_URL}/api/conversations/${encodeURIComponent(conversationId)}?include_skills=true`,
    { headers: sessionHeaders() },
  );
  expect(
    resp.ok(),
    `GET /api/conversations/${conversationId} returned ${resp.status()}`,
  ).toBe(true);
  return resp.json() as Promise<{
    agent?: {
      agent_context?: {
        skills?: unknown[];
        system_message_suffix?: string | null;
      };
    };
  }>;
}

function resolvedSkillSuffix(body: {
  agent?: {
    agent_context?: { system_message_suffix?: string | null };
  };
}): string {
  return body.agent?.agent_context?.system_message_suffix ?? "";
}

test.describe.configure({ mode: "serial" });

test.describe("Claude ACP Canvas skill overlay", () => {
  const conversationIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterAll(async ({ request, browser }) => {
    for (const id of conversationIds) {
      try {
        await deleteConversation(request, id);
      } catch {
        // best-effort
      }
    }
    const page = await browser.newPage();
    try {
      await seedLocalStorage(page);
      await ensureMockLLMAgentProfile(page.request);
      await resetToOpenHandsAgentViaUI(page);
      await ensureMockLLMProfile(page);
    } catch {
      // best-effort
    } finally {
      await page.close();
    }
    try {
      await resetMockLLM(request);
    } catch {
      // best-effort
    }
  });

  test("enabled openhands-automation instructions reach a Claude ACP session", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    await ensureMockLLMProfile(page);
    await ensureMockClaudeAcpAgentProfile(request);
    await patchSkillEnablement(request, {
      enabled_skills: ["openhands-automation"],
      disabled_skills: [],
    });

    let capturedPayload: Record<string, unknown> | null = null;
    const capturePayload = (req: import("@playwright/test").Request) => {
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
    page.on("request", capturePayload);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");
    await setChatInput(page, USER_MESSAGE);
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);
    page.off("request", capturePayload);

    const conversationId = getConversationIdFromURL(page);
    expect(conversationId).toBeTruthy();
    conversationIds.push(conversationId!);

    await test.step("POST carries the Claude skill launch overlay", async () => {
      expect(
        capturedPayload,
        "POST /api/conversations was not captured",
      ).not.toBeNull();
      expect(capturedPayload!.agent_profile_id).toBeTruthy();
      expect(capturedPayload!.agent_settings).toBeUndefined();
      const additions = capturedPayload!.agent_launch_additions as
        | { system_message_suffix_append?: string }
        | undefined;
      expect(additions?.system_message_suffix_append).toContain(
        AUTOMATION_HEADING,
      );
    });

    await test.step("resolved ACP conversation keeps the overlay after native skill strip", async () => {
      await expect
        .poll(
          async () => {
            const body = await readResolvedConversation(
              request,
              conversationId!,
            );
            expect(body.agent?.agent_context?.skills ?? []).toEqual([]);
            return resolvedSkillSuffix(body);
          },
          { timeout: 15_000 },
        )
        .toContain(AUTOMATION_HEADING);
    });

    await waitForNonUserMessageText(page, ACP_REPLY_TOKEN, 60_000);
  });

  test("disabling openhands-automation keeps those instructions out of a new Claude ACP session", async ({
    page,
    request,
  }) => {
    test.setTimeout(120_000);

    await ensureMockLLMProfile(page);
    await ensureMockClaudeAcpAgentProfile(request);
    await patchSkillEnablement(request, {
      enabled_skills: ["add-skill"],
      disabled_skills: ["openhands-automation"],
    });

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");
    await setChatInput(page, USER_MESSAGE);
    await page.getByTestId("submit-button").click();
    await waitForPath(page, /\/conversations\/.+/, 30_000);

    const conversationId = getConversationIdFromURL(page);
    expect(conversationId).toBeTruthy();
    conversationIds.push(conversationId!);

    const body = await readResolvedConversation(request, conversationId!);
    expect(resolvedSkillSuffix(body)).not.toContain(AUTOMATION_HEADING);

    await waitForNonUserMessageText(page, ACP_REPLY_TOKEN, 60_000);
  });
});
