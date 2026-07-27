/**
 * Mock-LLM E2E test: Create a cron automation, dispatch a run, and verify
 * the run completes with a conversation link.
 *
 * Exercises the full automation lifecycle end-to-end:
 *
 *   1. Setup: configure mock LLM profile and register a scripted trajectory
 *      whose terminal tool calls hit the REAL automation backend (running
 *      inside the bin/agent-canvas.mjs stack). The trajectory includes extra
 *      responses for the automation run's spawned conversation so it can
 *      finish and report COMPLETED.
 *   2. Conversation: type a prompt in the home chat launcher → mock LLM
 *      returns curl commands that create a cron automation and dispatch a run
 *      via the real automation API (through the ingress). Verify the run
 *      reaches COMPLETED status and has a conversation_id.
 *   3. UI verification: navigate to the /automations list page, click through
 *      to the automation detail page, verify the run shows COMPLETED with a
 *      clickable conversation link, and click through to verify the link
 *      navigates to the correct conversation page.
 *
 * No mock automation server is used — the real automation backend started by
 * bin/agent-canvas.mjs handles all /api/automation/* requests. The agent's
 * terminal commands authenticate with X-Session-API-Key header using the
 * stack's session API key.
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
  getMockLLMRequests,
} from "../utils/mock-llm-helpers";

// Token the test asserts on in the agent's text reply (step 2).
// The terminal printf breadcrumbs below are NOT asserted — they exist
// purely for log readability when debugging failures.
const AUTOMATION_REPLY_TOKEN = "MOCK_AUTOMATION_REPLY_OK";

const AUTOMATION_NAME = "Hello World Cron";
const CRON_SCHEDULE = "0 9 * * *";

// The ingress URL reachable from the agent's terminal. The agent-server
// Auth via X-Session-API-Key header (matching frontend automation-service.api.ts).
const AUTOMATION_API_BASE = `${BACKEND_URL}/api/automation/v1`;

/**
 * List automations from the real automation backend via the ingress.
 * Retries on 502/503 as a belt-and-suspenders safety net — even though the
 * Playwright webServer health check now probes /api/automation/v1, a brief
 * race between backend startup and the first test request is still possible.
 */
async function listAutomations(
  request: import("@playwright/test").APIRequestContext,
  retries = 15,
) {
  let lastStatus = 0;
  for (let i = 0; i < retries; i++) {
    const resp = await request.get(`${AUTOMATION_API_BASE}`, {
      headers: {
        "X-Session-API-Key": SESSION_API_KEY,
      },
    });
    lastStatus = resp.status();
    if (resp.ok()) return resp.json();
    // 502 = ingress can't reach the automation backend yet; retry
    if (lastStatus === 502 || lastStatus === 503) {
      await new Promise((r) => setTimeout(r, 2_000));
      continue;
    }
    // Any other error is unexpected
    break;
  }
  throw new Error(
    `GET automations returned ${lastStatus} after ${retries} retries`,
  );
}

/**
 * List runs for a specific automation via the real automation backend.
 */
async function listAutomationRuns(
  request: import("@playwright/test").APIRequestContext,
  automationId: string,
) {
  const resp = await request.get(
    `${AUTOMATION_API_BASE}/${encodeURIComponent(automationId)}/runs`,
    {
      headers: {
        "X-Session-API-Key": SESSION_API_KEY,
      },
    },
  );
  if (!resp.ok()) {
    const status = resp.status();
    // 502/503 = backend still starting — return empty so the caller retries
    if (status === 502 || status === 503) return { runs: [], items: [] };
    // Any other error is unexpected — surface it immediately
    throw new Error(
      `GET automation runs returned ${status} for ${automationId}`,
    );
  }
  return resp.json();
}

/**
 * Poll until a run reaches the expected status or times out.
 */
async function waitForRunStatus(
  request: import("@playwright/test").APIRequestContext,
  automationId: string,
  expectedStatus: string,
  timeoutMs = 30_000,
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const data = await listAutomationRuns(request, automationId);
    const runs = data.runs ?? data.items ?? [];
    const match = runs.find(
      (r: { status: string }) => r.status === expectedStatus,
    );
    if (match) return match;
    await new Promise((r) => setTimeout(r, 1_000));
  }
  throw new Error(
    `No run with status "${expectedStatus}" after ${timeoutMs}ms`,
  );
}

/**
 * Delete an automation (best-effort cleanup).
 */
async function deleteAutomation(
  request: import("@playwright/test").APIRequestContext,
  automationId: string,
) {
  await request.delete(
    `${AUTOMATION_API_BASE}/${encodeURIComponent(automationId)}`,
    {
      headers: {
        "X-Session-API-Key": SESSION_API_KEY,
      },
    },
  );
}

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM automation lifecycle", () => {
  const conversationIds = new Set<string>();
  const automationIds = new Set<string>();
  /** conversation_id from the completed automation run (set in step 2, verified in step 3) */
  let runConversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterEach(async ({ page, request }) => {
    // Collect conversation IDs from the URL for cleanup
    const match = page.url().match(/\/conversations\/([^/?#]+)/);
    if (match?.[1]) conversationIds.add(decodeURIComponent(match[1]));

    // Clean up conversations but NOT automations — step 3 needs the
    // automation created in step 2 to verify it appears on the page.
    for (const id of Array.from(conversationIds)) {
      try {
        await deleteConversation(request, id);
        conversationIds.delete(id);
      } catch {
        // best-effort cleanup
      }
    }
  });

  // Safety net: delete any leftover automations and reset the mock LLM.
  // Resetting here (instead of afterEach) preserves named trajectories
  // registered in step 1 for activation in step 2.
  test.afterAll(async ({ request }) => {
    for (const id of Array.from(automationIds)) {
      try {
        await deleteAutomation(request, id);
      } catch {
        // best-effort
      }
    }
    automationIds.clear();

    // Reset mock LLM so subsequent test suites start fresh.
    try {
      await resetMockLLM(request);
    } catch {
      // best-effort — the mock server may have already shut down
    }
  });

  // ── Step 1: Ensure LLM profile + register the automation trajectory ─

  test("step 1: setup LLM profile and register automation trajectory", async ({
    page,
    request,
  }) => {
    // Ensure the mock LLM profile is configured via the Settings UI
    await ensureMockLLMProfile(page);

    // Build the terminal commands the mock LLM will return.
    // The curl commands hit the REAL automation backend through the ingress.
    // Auth uses $OPENHANDS_AUTOMATION_API_KEY which the agent-server
    // exposes as an env var in the terminal sandbox.
    //
    // Turn 1: Create the automation via curl preset/prompt endpoint.
    // Turn 2: Extract the automation ID and dispatch a run.
    // Turn 3: Text reply with a verification token.

    // Auth: hardcode the session API key directly in the curl commands.
    // The agent-server terminal may not inherit all parent env vars (the SDK
    // sandboxes the execution environment), so $OPENHANDS_AUTOMATION_API_KEY
    // may not be available. Using the key directly is safe in a test context.
    const authHeader = `-H 'X-Session-API-Key: ${SESSION_API_KEY}'`;

    const createCmd = [
      `curl -s -X POST '${AUTOMATION_API_BASE}/preset/prompt'`,
      `-H 'Content-Type: application/json'`,
      authHeader,
      `-d '${JSON.stringify({
        name: AUTOMATION_NAME,
        prompt: "echo hello world",
        trigger: { type: "cron", schedule: CRON_SCHEDULE, timezone: "UTC" },
      })}'`,
      `-o /tmp/auto_result.json`,
      `-w '\\nHTTP_CODE:%{http_code}\\n'`,
      `&& cat /tmp/auto_result.json`,
      `&& printf 'AUTOMATION_CREATED\\n'`,
    ].join(" ");

    const dispatchCmd = [
      `AID=$(python3 -c "import json; print(json.load(open('/tmp/auto_result.json'))['id'])")`,
      `&& curl -s -X POST "${AUTOMATION_API_BASE}/$AID/dispatch"`,
      authHeader,
      `-H 'Content-Type: application/json'`,
      `-w '\\nHTTP_CODE:%{http_code}\\n'`,
      `&& printf 'AUTOMATION_DISPATCHED\\n'`,
    ].join(" ");

    // ⚠️  Padding response (index 0):
    // Public skills are bundled from @openhands/extensions at build time.
    // The agent-server's skill-activation pipeline makes one internal LLM
    // call to decide which skills to inject before the agent loop starts.
    // Our user message mentions "automation", which matches the
    // openhands-automation skill, triggering this internal call.
    // The conversation test does NOT need padding because its prompt
    // ("run this bash command") does not match any skill trigger.
    //
    // If the padding ever becomes misaligned (e.g. the agent-server stops
    // making this call or starts making two), step 2's
    // waitForNonUserMessageText(AUTOMATION_REPLY_TOKEN) will time out
    // quickly, making the failure obvious. See AGENTS.md → "Padding
    // response for internal LLM call" for more context.
    //
    // After the main conversation finishes (responses 0-3), the dispatched
    // automation run spawns a NEW conversation on the same agent-server.
    // That conversation also calls the mock LLM. We append extra text
    // responses (4-6) so the run's conversation can finish normally, the
    // script fires its completion callback, and the run reaches COMPLETED.
    await registerTrajectory(request, "automation-lifecycle", [
      // ── Main conversation (responses 0-3) ──
      { text: "" }, // 0: consumed by skill-activation LLM call (see above)
      {
        // 1: create the automation via curl
        tool_call: {
          name: "terminal",
          arguments: { command: createCmd },
        },
      },
      {
        // 2: dispatch a run via curl
        tool_call: {
          name: "terminal",
          arguments: { command: dispatchCmd },
        },
      },
      { text: AUTOMATION_REPLY_TOKEN }, // 3: finish main conversation

      // ── Automation run's conversation (responses 4+) ──
      // The run starts a fresh conversation with the automation prompt.
      // Provide enough responses for any internal LLM calls + the agent's
      // turn so the conversation finishes and the completion callback fires.
      { text: "" }, // 4: possible internal/condenser call
      { text: "Done. Hello world echoed successfully." }, // 5: agent reply
      { text: "" }, // 6: safety buffer for any follow-up internal call
    ]);

    // Activate it so the mock LLM uses this trajectory for the next conversation
    await activateTrajectory(request, "automation-lifecycle");
  });

  // ── Step 2: Create automation via conversation ─────────────────────

  test("step 2: create automation and dispatch run via the UI", async ({
    page,
    request,
  }) => {
    // Budget: ~150 s for the two dominant waits (waitForNonUserMessageText 60 s
    // + waitForRunStatus 90 s), plus ~30 s margin for navigation/page loads.
    // If CI proves flaky, bump to 240_000.
    test.setTimeout(180_000);
    // Re-activate in case the mock-LLM server restarted between steps (belt-and-suspenders)
    await activateTrajectory(request, "automation-lifecycle");

    await routeSessionApiKey(page);

    // Navigate to the home page and type a prompt to create the automation.
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    await test.step("type prompt and submit", async () => {
      await waitForTestId(page, "home-chat-launcher", 15_000);

      const userMessage =
        "Create a cron automation that echoes hello world every morning at 9am.";

      // Set contenteditable text via evaluate (contentEditable divs don't
      // respond reliably to Playwright's .fill() or .type()).
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
        { testId: "chat-input", text: userMessage },
      );

      await page.getByTestId("submit-button").click();
    });

    // Wait for navigation to the new conversation page
    await test.step("wait for conversation to start", async () => {
      await waitForPath(page, /\/conversations\/.+/, 30_000);
    });

    const conversationId = getConversationIdFromURL(page);
    conversationIds.add(conversationId);

    // ── Verify: the LLM reply token appears in the chat UI ──

    await test.step("verify LLM reply token in chat UI", async () => {
      await waitForNonUserMessageText(page, AUTOMATION_REPLY_TOKEN, 60_000);
    });

    // ── Verify: automation was created in the real automation backend ──

    await test.step("verify automation was created", async () => {
      const data = await listAutomations(request);
      const automations = data.automations ?? data.items ?? [];
      expect(
        automations.length,
        `Expected at least 1 automation, got: ${JSON.stringify(data).slice(0, 500)}`,
      ).toBeGreaterThanOrEqual(1);

      const created = automations.find(
        (a: { name: string }) => a.name === AUTOMATION_NAME,
      );
      expect(created, `Automation "${AUTOMATION_NAME}" not found`).toBeTruthy();
      automationIds.add(created.id);
      expect(created.trigger?.schedule).toBe(CRON_SCHEDULE);
      expect(created.enabled).toBe(true);
    });

    // ── Verify: run completed successfully with a conversation link ──

    await test.step("verify run completed with conversation link", async () => {
      const data = await listAutomations(request);
      const automations = data.automations ?? data.items ?? [];
      const automation = automations.find(
        (a: { name: string }) => a.name === AUTOMATION_NAME,
      );
      expect(automation, "Automation should exist for run check").toBeTruthy();
      automationIds.add(automation.id);

      // Wait for the run to reach COMPLETED. The trajectory includes extra
      // responses (indices 4-6) for the automation run's spawned conversation
      // so it can finish and fire the completion callback.
      const run = await waitForRunStatus(
        request,
        automation.id,
        "COMPLETED",
        90_000,
      );
      expect(run.conversation_id).toBeTruthy();
      // Store the conversation ID for the click-through verification in step 3
      runConversationId = run.conversation_id;
    });

    // ── Verify: no error banners ──

    await test.step("verify no error banners", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });

    // ── Verify: runtime services info is included in LLM requests ──

    await test.step("verify runtime services info in LLM system prompt", async () => {
      const llmRequests = await getMockLLMRequests(request);
      expect(
        llmRequests.length,
        "mock LLM should have received at least one completion request",
      ).toBeGreaterThan(0);

      // The <RUNTIME_SERVICES> block should appear in a system message
      // sent to the LLM. Walk all captured requests looking for it.
      function findRuntimeServicesContent(
        reqs: Record<string, unknown>[],
      ): string | null {
        for (const req of reqs) {
          const messages = req.messages as
            | Array<{ role: string; content: unknown }>
            | undefined;
          if (!Array.isArray(messages)) continue;
          for (const msg of messages) {
            if (msg.role !== "system") continue;
            const text =
              typeof msg.content === "string"
                ? msg.content
                : Array.isArray(msg.content)
                  ? (msg.content as Array<{ type?: string; text?: string }>)
                      .filter((c) => c.type === "text" || typeof c === "string")
                      .map((c) => (typeof c === "string" ? c : c.text ?? ""))
                      .join("")
                  : "";
            if (text.includes("<RUNTIME_SERVICES>")) return text;
          }
        }
        return null;
      }

      const runtimeBlock = findRuntimeServicesContent(llmRequests);
      expect(
        runtimeBlock,
        `Expected <RUNTIME_SERVICES> block in a system message.\n` +
          `Received ${llmRequests.length} LLM request(s) but none contained it.`,
      ).toBeTruthy();

      // Verify the block includes key services that should be present
      // in the full agent-canvas stack (agent-server + automation + ingress).
      expect(runtimeBlock).toContain("Agent Server");
      expect(runtimeBlock).toContain("Automation backend");
      expect(runtimeBlock).toContain("/api/automation");
      expect(runtimeBlock).toContain("</RUNTIME_SERVICES>");
    });
  });

  // ── Step 3: Verify automation on list page, click through to detail, verify run link ─

  test("step 3: verify automation and run on the automations page", async ({
    page,
    request,
  }) => {
    test.setTimeout(60_000);
    await routeSessionApiKey(page);
    await page.goto("/automations", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);

    await test.step("automation card visible on list page", async () => {
      await waitForTestId(page, "automations-add-automation", 15_000);

      // Wait for the automation name to appear on the page (the list
      // may take a moment to load from the automation backend).
      await expect(page.getByText(AUTOMATION_NAME)).toBeVisible({
        timeout: 15_000,
      });
    });

    await test.step("click through to automation detail page", async () => {
      // The automation name is a link — clicking it navigates to /automations/:id
      await page.getByText(AUTOMATION_NAME).click();
      await waitForPath(page, /\/automations\/.+/, 10_000);
    });

    // Verify the automation shows an "Active" enabled-status badge on the detail page
    await test.step("verify automation shows active status badge", async () => {
      const activeBadge = page.getByTestId("active-status-badge-active");
      await expect(activeBadge).toBeVisible({ timeout: 10_000 });
    });

    // Verify the cron schedule is displayed in the configuration section.
    // The ConfigurationSection renders schedule_human (e.g. "Every day at 9:00 AM")
    // or falls back to the raw cron expression.
    await test.step("verify cron schedule displayed on detail page", async () => {
      await expect(page.getByText(CRON_SCHEDULE)).toBeVisible({ timeout: 10_000 });
    });

    await test.step("verify run shows COMPLETED with conversation link", async () => {
      // The activity log should show a COMPLETED badge (translated as "Successful")
      const completedIcon = page.getByTestId("run-status-icon-completed");
      await expect(completedIcon).toBeVisible({ timeout: 15_000 });

      // Verify step 2 populated runConversationId — without it the
      // click-through assertion below would silently pass.
      expect(
        runConversationId,
        "step 2 must set runConversationId before step 3 can verify the link",
      ).toBeTruthy();
      if (runConversationId) {
        const runLinks = page.locator(
          `a[href="/conversations/${runConversationId}"]`,
        );
        // There may be multiple matching links (e.g. header + activity row);
        // assert at least one exists and click the first.
        await expect(runLinks.first()).toBeVisible({ timeout: 10_000 });

        // Click the run link and verify it navigates to the conversation page
        await runLinks.first().click();
        await waitForPath(page, /\/conversations\/.+/, 10_000);
        expect(page.url()).toContain(runConversationId);
      }
    });

    // Clean up automations at the end of the last test
    await test.step("cleanup automations", async () => {
      for (const id of Array.from(automationIds)) {
        try {
          await deleteAutomation(request, id);
          automationIds.delete(id);
        } catch {
          // best-effort
        }
      }
    });
  });

});
