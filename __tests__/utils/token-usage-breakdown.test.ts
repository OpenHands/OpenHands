import { describe, expect, it } from "vitest";
import type {
  RuntimeConversationStats,
  RuntimeMetrics,
  RuntimeTokenUsage,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { createOtherActionEvent } from "../../test-utils";
import { buildTokenUsageBreakdown } from "#/utils/token-usage-breakdown";

const call = (
  responseId: string,
  promptTokens: number,
  completionTokens: number,
): RuntimeTokenUsage => ({
  response_id: responseId,
  prompt_tokens: promptTokens,
  completion_tokens: completionTokens,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  context_window: 200_000,
  per_turn_token: promptTokens + completionTokens,
});

const metrics = (tokenUsages: RuntimeTokenUsage[]): RuntimeMetrics => ({
  model_name: "test-model",
  accumulated_cost: 0,
  max_budget_per_task: null,
  accumulated_token_usage: null,
  costs: [],
  response_latencies: [],
  token_usages: tokenUsages,
});

const toolCall = (
  id: string,
  toolName: string,
  responseId: string,
): OpenHandsEvent => ({
  ...createOtherActionEvent(id),
  tool_name: toolName,
  llm_response_id: responseId,
});

const agentMessage = (id: string, responseId: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2026-10-06T00:00:00Z",
    source: "agent",
    llm_message: { role: "assistant", content: [] },
    activated_skills: [],
    extended_content: [],
    llm_response_id: responseId,
  }) as OpenHandsEvent;

describe("buildTokenUsageBreakdown", () => {
  it("groups each call's exact usage under the activity its events record", () => {
    // Arrange: two terminal calls, one parallel terminal + file_editor call,
    // and a final reply to the user.
    const stats: RuntimeConversationStats = {
      usage_to_metrics: {
        default: metrics([
          call("r1", 1_000, 50),
          call("r2", 2_000, 70),
          call("r3", 3_000, 400),
          call("r4", 4_000, 30),
        ]),
      },
    };
    const events = [
      toolCall("a1", "terminal", "r1"),
      toolCall("a2", "terminal", "r2"),
      toolCall("a3", "terminal", "r3"),
      toolCall("a4", "file_editor", "r3"),
      agentMessage("m1", "r4"),
    ];

    // Act
    const breakdown = buildTokenUsageBreakdown(stats, events);

    // Assert: one row per activity, largest first, nothing split or lost.
    expect(breakdown.rows).toEqual([
      expect.objectContaining({
        key: "agent_message",
        calls: 1,
        inputTokens: 4_000,
        outputTokens: 30,
        totalTokens: 4_030,
      }),
      expect.objectContaining({
        key: "tools:file_editor+terminal",
        activity: { kind: "tools", toolNames: ["file_editor", "terminal"] },
        calls: 1,
        totalTokens: 3_400,
      }),
      expect.objectContaining({
        key: "tools:terminal",
        calls: 2,
        inputTokens: 3_000,
        outputTokens: 120,
        totalTokens: 3_120,
      }),
    ]);
    expect(breakdown.totalCalls).toBe(4);
    expect(breakdown.totalTokens).toBe(10_550);
    expect(breakdown.unmatchedCalls).toBe(0);
  });

  it("labels calls that no event explains by the LLM that made them", () => {
    // Arrange: the agent LLM has one matched and one unmatched call; the
    // condenser and a hook LLM never produce matching events.
    const stats: RuntimeConversationStats = {
      usage_to_metrics: {
        default: metrics([call("r1", 100, 10), call("r-failed", 200, 0)]),
        condenser: metrics([call("c1", 300, 40)]),
        "prompt-hook:lint": metrics([call("h1", 50, 5)]),
      },
    };
    const events = [toolCall("a1", "terminal", "r1")];

    // Act
    const breakdown = buildTokenUsageBreakdown(stats, events);

    // Assert
    expect(breakdown.rows.map((row) => row.activity)).toEqual([
      { kind: "condensation" },
      { kind: "unattributed" },
      { kind: "tools", toolNames: ["terminal"] },
      { kind: "other_llm", usageId: "prompt-hook:lint" },
    ]);
    // The condenser never has events, so only the agent and hook calls can
    // still be matched by loading older history.
    expect(breakdown.unmatchedCalls).toBe(2);
  });
});
