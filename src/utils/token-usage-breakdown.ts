import type { RuntimeConversationStats } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isCondensationEvent,
  isMessageEvent,
} from "#/types/agent-server/type-guards";

/** The `usage_id` the SDK gives the condenser's LLM. */
export const CONDENSER_USAGE_ID = "condenser";

/**
 * What a group of LLM calls did. Calls are matched to activities through the
 * `llm_response_id` that the agent's events carry, so every call belongs to
 * exactly one activity and no token count is estimated or split.
 */
export type TokenUsageActivity =
  /** Calls that produced tool calls; parallel calls keep every tool name. */
  | { kind: "tools"; toolNames: string[] }
  /** Calls that produced a message to the user instead of a tool call. */
  | { kind: "agent_message" }
  /** Calls that summarized the history (condenser). */
  | { kind: "condensation" }
  /** Calls from the agent's LLM with no matching event (e.g. a failed call). */
  | { kind: "unattributed" }
  /** Calls from another LLM of the conversation (hooks, vision, …). */
  | { kind: "other_llm"; usageId: string };

export interface TokenUsageActivityRow {
  /** Unique, stable key: one row per activity. */
  key: string;
  activity: TokenUsageActivity;
  calls: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  /** Input plus output tokens, the same total the Usage tab shows. */
  totalTokens: number;
}

export interface TokenUsageBreakdown {
  /** Sorted by `totalTokens`, largest first. */
  rows: TokenUsageActivityRow[];
  totalCalls: number;
  totalTokens: number;
  /**
   * Calls that no event matched, excluding the condenser. A positive count
   * means older events can still attribute them, so callers load the full
   * history before they show the result as final.
   */
  unmatchedCalls: number;
}

type ResponseActivity = Exclude<
  TokenUsageActivity,
  { kind: "unattributed" } | { kind: "other_llm" }
>;

const readLlmResponseId = (event: OpenHandsEvent): string | null =>
  "llm_response_id" in event &&
  typeof event.llm_response_id === "string" &&
  event.llm_response_id.length > 0
    ? event.llm_response_id
    : null;

/**
 * Index the events by the LLM response that produced them. A response with
 * parallel tool calls yields several action events with one response id.
 */
function indexResponseActivities(
  events: readonly OpenHandsEvent[],
): Map<string, ResponseActivity> {
  const toolNamesByResponse = new Map<string, Set<string>>();
  const otherByResponse = new Map<string, ResponseActivity>();

  events.forEach((event) => {
    const responseId = readLlmResponseId(event);
    if (!responseId) return;

    if (isActionEvent(event)) {
      const toolNames = toolNamesByResponse.get(responseId) ?? new Set();
      toolNames.add(event.tool_name);
      toolNamesByResponse.set(responseId, toolNames);
    } else if (isCondensationEvent(event)) {
      otherByResponse.set(responseId, { kind: "condensation" });
    } else if (isMessageEvent(event) && event.source === "agent") {
      otherByResponse.set(responseId, { kind: "agent_message" });
    }
  });

  const activities = new Map<string, ResponseActivity>(otherByResponse);
  toolNamesByResponse.forEach((toolNames, responseId) => {
    activities.set(responseId, {
      kind: "tools",
      toolNames: [...toolNames].sort(),
    });
  });
  return activities;
}

const getActivityKey = (activity: TokenUsageActivity): string => {
  switch (activity.kind) {
    case "tools":
      return `tools:${activity.toolNames.join("+")}`;
    case "other_llm":
      return `other_llm:${activity.usageId}`;
    default:
      return activity.kind;
  }
};

/**
 * Group the per-call token usage records of a conversation by the activity
 * that each call produced. Totals equal the conversation's accumulated usage.
 */
export function buildTokenUsageBreakdown(
  stats: RuntimeConversationStats,
  events: readonly OpenHandsEvent[],
): TokenUsageBreakdown {
  const activities = indexResponseActivities(events);
  const usageEntries = Object.entries(stats.usage_to_metrics ?? {});

  // A usage id with at least one matched call belongs to the agent's LLM, so
  // its other calls are unattributed rather than another LLM's work.
  const agentUsageIds = new Set(
    usageEntries
      .filter(([, metrics]) =>
        (metrics.token_usages ?? []).some(
          (usage) => !!usage.response_id && activities.has(usage.response_id),
        ),
      )
      .map(([usageId]) => usageId),
  );

  const rowsByKey = new Map<string, TokenUsageActivityRow>();
  let unmatchedCalls = 0;

  usageEntries.forEach(([usageId, metrics]) => {
    (metrics.token_usages ?? []).forEach((usage) => {
      const matched = usage.response_id
        ? activities.get(usage.response_id)
        : undefined;
      let activity: TokenUsageActivity;
      if (matched) {
        activity = matched;
      } else if (usageId === CONDENSER_USAGE_ID) {
        activity = { kind: "condensation" };
      } else {
        unmatchedCalls += 1;
        activity = agentUsageIds.has(usageId)
          ? { kind: "unattributed" }
          : { kind: "other_llm", usageId };
      }

      const key = getActivityKey(activity);
      const row = rowsByKey.get(key) ?? {
        key,
        activity,
        calls: 0,
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        totalTokens: 0,
      };
      const inputTokens = usage.prompt_tokens ?? 0;
      const outputTokens = usage.completion_tokens ?? 0;
      row.calls += 1;
      row.inputTokens += inputTokens;
      row.outputTokens += outputTokens;
      row.cacheReadTokens += usage.cache_read_tokens ?? 0;
      row.totalTokens += inputTokens + outputTokens;
      rowsByKey.set(key, row);
    });
  });

  const rows = [...rowsByKey.values()].sort(
    (a, b) => b.totalTokens - a.totalTokens || a.key.localeCompare(b.key),
  );

  return {
    rows,
    totalCalls: rows.reduce((sum, row) => sum + row.calls, 0),
    totalTokens: rows.reduce((sum, row) => sum + row.totalTokens, 0),
    unmatchedCalls,
  };
}
