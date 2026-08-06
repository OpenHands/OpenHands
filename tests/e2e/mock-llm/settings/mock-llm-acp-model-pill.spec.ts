/**
 * Mock-LLM E2E test: ACP dynamic model + effort pill (agent-canvas M6a).
 *
 * Exercises the chat-input model pill (`ChatInputModel` /
 * `useChatInputModelState`) end-to-end against a *real* agent-server driving
 * the mock ACP agent (`mock-acp-server.py`), which — from M6a — advertises a
 * `session/new` `configOptions` "model" select (3 mock models) and "effort"
 * select (claude-code's static levels), and handles
 * `session/set_config_option` for both:
 *
 *   1. Configure the "default" agent profile as ACP (`acp_server:
 *      "claude-code"`) pointed at the mock ACP server, directly through the
 *      agent-profiles API (see `ensureAcpClaudeCodeMockProfile`'s docstring
 *      for why the Settings → Agent UI can't produce this combination).
 *   2. Start a conversation from the home page and verify the ACP agent
 *      responds — this also drives the ACP session through `session/new`,
 *      which is what populates `ConversationInfo.available_models` /
 *      `current_model_id` / `current_effort` / `available_efforts` from the
 *      mock's configOptions.
 *   3. Open the model pill and verify the LIVE mock-advertised models (their
 *      display names) appear — proving
 *      `ConversationInfo.available_models` → pill end-to-end.
 *   4. Switch to another model via the pill and verify the pill/current
 *      selection updates after the refetch.
 *   5. Verify the effort section renders claude-code's static levels and
 *      switch effort to "max"; verify the pill reflects the composite
 *      "<model> · max" label and the effort row's checkmark moves.
 *
 * A custom-server (acp_server: "custom") angle was considered but skipped:
 * it would need a second profile fixture (custom-preset provider has no
 * curated model list of its own and no effort UI at all —
 * `getAcpEffortLevels` returns `null` for it — so the only thing left to
 * prove is the live-models list, which this spec already covers for
 * claude-code). Not worth the extra profile-lifecycle complexity for this
 * pass.
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
  ensureAcpClaudeCodeMockProfile,
  setChatInput,
  BACKEND_URL,
  SESSION_API_KEY,
} from "../utils/mock-llm-helpers";

const USER_MESSAGE = "Hello ACP agent, please reply.";

// Mirrors MOCK_MODELS / EFFORT_LEVELS in
// tests/e2e/mock-llm/scripts/mock-acp-server.py — keep in sync.
const MOCK_MODELS = [
  { id: "mock-fast", label: "Mock Fast" },
  { id: "mock-smart", label: "Mock Smart" },
  { id: "mock-deep", label: "Mock Deep" },
] as const;
const DEFAULT_MODEL_ID = "mock-smart";
const SWITCH_TARGET_MODEL_ID = "mock-deep";
// The mock server advertises currentValue "high" for its effort select;
// whether the UI sees it depends on the agent-server (see the liveEffort
// probe in step 3), so assertions derive from the probe, not this value.
const EFFORT_LEVELS = [
  "default",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

// labelForAcpModel (src/constants/acp-providers.ts) joins a parsed
// "<base>/<effort>" id as "<baseLabel> · <effort>" — literal middle dot.
const DOT = "·";

test.describe.configure({ mode: "serial" });

test.describe("mock-LLM ACP model pill (dynamic model + effort switching)", () => {
  let conversationId: string | null = null;

  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
  });

  test.afterAll(async ({ request, browser }) => {
    // Clean up the conversation
    if (conversationId) {
      try {
        await deleteConversation(request, conversationId);
      } catch {
        // best-effort
      }
    }

    // Restore the "default" agent profile back to OpenHands + the mock LLM
    // profile so subsequent test suites (which expect agent_kind=openhands)
    // are not affected by our ACP configuration. Mirrors
    // mock-llm-acp-agent.spec.ts's cleanup exactly.
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

  // ── Step 1: Configure the ACP profile directly via the API ──────────

  test("step 1: configure ACP agent profile (claude-code, mock server command)", async ({
    page,
  }) => {
    // The agent-server may make internal LLM calls (condenser) even for ACP
    // conversations. Ensure a mock LLM profile exists first so those calls
    // don't fail — same rationale as mock-llm-acp-agent.spec.ts. This MUST
    // run before ensureAcpClaudeCodeMockProfile: ensureMockLLMProfile itself
    // (re)activates the "default" agent profile as agent_kind="openhands"
    // as its last step, which would clobber the ACP configuration if run
    // afterward.
    await ensureMockLLMProfile(page);

    await ensureAcpClaudeCodeMockProfile(page.request);

    // ── Verify: agent-profiles API reflects the ACP configuration ──
    const resp = await page.request.get(
      `${BACKEND_URL}/api/agent-profiles/default`,
      { headers: { "X-Session-API-Key": SESSION_API_KEY } },
    );
    expect(
      resp.ok(),
      `GET /api/agent-profiles/default returned ${resp.status()}`,
    ).toBe(true);
    const detail = await resp.json();
    const profile = detail?.profile as Record<string, unknown>;
    expect(profile?.agent_kind).toBe("acp");
    expect(profile?.acp_server).toBe("claude-code");
    expect(
      typeof profile?.acp_command === "string" &&
        (profile.acp_command as string).includes("mock-acp-server.py"),
      `acp_command should reference mock-acp-server.py, got: ${JSON.stringify(profile?.acp_command)}`,
    ).toBe(true);
  });

  // ── Step 2: Start an ACP conversation from the home page ────────────

  test("step 2: start ACP conversation and verify agent reply", async ({
    page,
  }) => {
    test.setTimeout(60_000);

    await routeSessionApiKey(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "home-chat-launcher");

    await setChatInput(page, USER_MESSAGE);
    await page.getByTestId("submit-button").click();

    await waitForPath(page, /\/conversations\/.+/, 30_000);
    conversationId = getConversationIdFromURL(page);

    // Session creation (session/new) happens synchronously as part of
    // starting the conversation, before the mock agent even sees this
    // prompt — but waiting for the reply token is the simplest robust
    // signal that the ACP subprocess is fully up and the session exists,
    // matching mock-llm-acp-agent.spec.ts's synchronization point.
    await waitForNonUserMessageText(page, ACP_REPLY_TOKEN, 30_000);
  });

  // ── Step 3: Dynamic model pill + effort switching ────────────────────

  test("step 3: model pill shows live models and effort levels, and switches both", async ({
    page,
  }) => {
    test.skip(!conversationId, "step 2 must complete first");
    test.setTimeout(60_000);

    await routeSessionApiKey(page);
    await page.goto(`/conversations/${conversationId}`, {
      waitUntil: "domcontentloaded",
    });
    await dismissAnalyticsModal(page);
    await waitForTestId(page, "chat-interface", 30_000);

    const pill = page.getByTestId("chat-input-llm-model");
    const popover = page.getByTestId("chat-input-llm-model-popover");
    await expect(pill).toBeVisible({ timeout: 15_000 });

    // Adaptive to the agent-server in use: servers with the effort-surfacing
    // changes report the session's live effort ("high", the mock's advertised
    // currentValue) on ConversationInfo; stock servers omit the field and the
    // pill falls back to "default" until an effort is explicitly chosen
    // (use-chat-input-model-state's parse-fallback). Probe once and assert
    // the matching behavior so this spec passes against both.
    const infoResp = await page.request.get(
      `${BACKEND_URL}/api/conversations/${conversationId}`,
      { headers: { "X-Session-API-Key": SESSION_API_KEY } },
    );
    expect(infoResp.ok()).toBe(true);
    const info = (await infoResp.json()) as { current_effort?: string | null };
    const liveEffort =
      typeof info.current_effort === "string" && info.current_effort !== ""
        ? info.current_effort
        : null;
    const initialEffort = liveEffort ?? "default";

    await test.step("pill starts on the mock server's default model", async () => {
      // A bare model id (no live effort embedded — labelForAcpModel only
      // suffixes an effort parsed out of the id itself, and the session's
      // freshly-reported current_model_id has no "/" in it).
      await expect(pill).toHaveAttribute("title", DEFAULT_MODEL_ID, {
        timeout: 15_000,
      });
    });

    await test.step("opening the pill lists every live mock-advertised model", async () => {
      await pill.click();
      await expect(popover).toBeVisible({ timeout: 5_000 });

      for (const model of MOCK_MODELS) {
        const row = page.getByTestId(`chat-input-acp-model-option-${model.id}`);
        await expect(row).toBeVisible({ timeout: 5_000 });
        await expect(row).toContainText(model.label);
      }

      // The default model is highlighted (checkmark) as current.
      await expect(
        page
          .getByTestId(`chat-input-acp-model-option-${DEFAULT_MODEL_ID}`)
          .locator("svg"),
      ).toBeVisible();
    });

    await test.step("the effort section renders claude-code's static levels", async () => {
      for (const effort of EFFORT_LEVELS) {
        await expect(
          page.getByTestId(`chat-input-acp-effort-option-${effort}`),
        ).toBeVisible({ timeout: 5_000 });
      }

      // Live-effort servers highlight the mock's advertised default
      // ("high"); stock servers highlight the "default" fallback row.
      await expect(
        page
          .getByTestId(`chat-input-acp-effort-option-${initialEffort}`)
          .locator("svg"),
      ).toBeVisible();
    });

    await test.step("switching to another model updates the pill", async () => {
      await page
        .getByTestId(`chat-input-acp-model-option-${SWITCH_TARGET_MODEL_ID}`)
        .click();
      await expect(popover).not.toBeVisible({ timeout: 5_000 });

      // The switch composes the base model with the session's current
      // effort — composeAcpModelId/M5. With a live-effort server that is
      // the mock's "high" and the pill's title becomes the composite
      // "<label> · <effort>"; on stock servers the current effort is the
      // "default" fallback, which composes to the bare id. labelForAcpModel
      // resolves <label> against claude-code's *static* curated registry
      // (real Claude model ids), which our mock ids aren't in, so it falls
      // back to the raw id ("mock-deep") rather than the live-advertised
      // display name ("Mock Deep").
      const expectedTitle =
        liveEffort && liveEffort !== "default"
          ? `${SWITCH_TARGET_MODEL_ID} ${DOT} ${liveEffort}`
          : SWITCH_TARGET_MODEL_ID;
      await expect(pill).toHaveAttribute("title", expectedTitle, {
        timeout: 15_000,
      });
    });

    await test.step("reopening the pill shows the switched model as current", async () => {
      await pill.click();
      await expect(popover).toBeVisible({ timeout: 5_000 });
      await expect(
        page
          .getByTestId(`chat-input-acp-model-option-${SWITCH_TARGET_MODEL_ID}`)
          .locator("svg"),
      ).toBeVisible();
      // The effort highlight is unaffected by a model-only switch.
      await expect(
        page
          .getByTestId(`chat-input-acp-effort-option-${initialEffort}`)
          .locator("svg"),
      ).toBeVisible();
    });

    if (liveEffort) {
      // Applying an effort requires the server-side composite splitter,
      // which ships with the same server changes that surface live effort —
      // gate on the probe. A stock server would (correctly) reject the
      // composite id: the mock validates model values like claude-agent-acp
      // does, so the switch 400s and the pill stays on the bare id.
      await test.step("switching effort to max updates the pill", async () => {
        await page.getByTestId("chat-input-acp-effort-option-max").click();
        await expect(popover).not.toBeVisible({ timeout: 5_000 });

        await expect(pill).toHaveAttribute(
          "title",
          `${SWITCH_TARGET_MODEL_ID} ${DOT} max`,
          { timeout: 15_000 },
        );
      });

      await test.step("reopening the pill shows max as current with the model unchanged", async () => {
        await pill.click();
        await expect(popover).toBeVisible({ timeout: 5_000 });
        await expect(
          page.getByTestId("chat-input-acp-effort-option-max").locator("svg"),
        ).toBeVisible();
        await expect(
          page
            .getByTestId(
              `chat-input-acp-model-option-${SWITCH_TARGET_MODEL_ID}`,
            )
            .locator("svg"),
        ).toBeVisible();
      });
    } else {
      await test.step("effort switching skipped (server reports no live effort) — close the pill", async () => {
        // The popover closes on toggle/click-outside, not Escape.
        await pill.click();
        await expect(popover).not.toBeVisible({ timeout: 5_000 });
      });
    }

    await test.step("no error banner after switching model and effort", async () => {
      const errorBanner = page.getByTestId("error-message-banner");
      await expect(errorBanner).not.toBeVisible({ timeout: 2_000 });
    });
  });
});
