import { describe, expect, it } from "vitest";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  deriveSubagents,
  getActivityMetrics,
  getActivityStatusDescriptor,
  mergeActivityTail,
  pickLatestActivity,
  selectActiveConversations,
} from "#/components/features/activity/activity-view-model";

/**
 * A task action's event `id` and its `tool_call_id` are distinct values on the
 * wire, and its observation references the action through `action_id` (the
 * event id), not the tool-call id. Defaults here model that so the pairing is
 * exercised realistically; callers can force a collision.
 */
const taskAction = (
  toolCallId: string,
  subagentType: string,
  eventId: string = `action-${toolCallId}`,
): OpenHandsEvent =>
  ({
    id: eventId,
    timestamp: "2026-10-06T00:00:00Z",
    source: "agent",
    action: { kind: "TaskAction", subagent_type: subagentType },
    tool_name: "task",
    tool_call_id: toolCallId,
  }) as unknown as OpenHandsEvent;

const taskObservation = (
  actionId: string,
  isError: boolean,
  toolCallId: string = actionId,
): OpenHandsEvent =>
  ({
    id: `observation-${actionId}`,
    timestamp: "2026-10-06T00:00:01Z",
    source: "environment",
    action_id: actionId,
    tool_name: "task",
    tool_call_id: toolCallId,
    observation: { kind: "TaskObservation", is_error: isError },
  }) as unknown as OpenHandsEvent;

const bashAction = (toolCallId: string): OpenHandsEvent =>
  ({
    id: `action-${toolCallId}`,
    timestamp: "2026-10-06T00:00:00Z",
    source: "agent",
    action: { kind: "ExecuteBashAction", command: "npm test" },
    tool_name: "execute_bash",
    tool_call_id: toolCallId,
  }) as unknown as OpenHandsEvent;

const assistantMessage = (text: string): OpenHandsEvent =>
  ({
    id: `message-${text}`,
    timestamp: "2026-10-06T00:00:02Z",
    source: "agent",
    llm_message: {
      role: "assistant",
      content: [{ type: "text", text }],
    },
  }) as unknown as OpenHandsEvent;

const conversation = (overrides: Partial<AppConversation>): AppConversation =>
  ({
    id: "c1",
    title: "Conversation",
    execution_status: ExecutionStatus.RUNNING,
    metrics: null,
    ...overrides,
  }) as AppConversation;

// @spec LAV-002 — A row conveys status, current step, and spend
describe("getActivityStatusDescriptor", () => {
  it("flags only states that require user action as needing attention", () => {
    expect(
      getActivityStatusDescriptor(ExecutionStatus.WAITING_FOR_CONFIRMATION)
        .needsAttention,
    ).toBe(true);
    expect(
      getActivityStatusDescriptor(ExecutionStatus.ERROR).needsAttention,
    ).toBe(true);
    expect(
      getActivityStatusDescriptor(ExecutionStatus.STUCK).needsAttention,
    ).toBe(true);

    expect(
      getActivityStatusDescriptor(ExecutionStatus.RUNNING).needsAttention,
    ).toBe(false);
    expect(
      getActivityStatusDescriptor(ExecutionStatus.PAUSED).needsAttention,
    ).toBe(false);
    expect(
      getActivityStatusDescriptor(ExecutionStatus.FINISHED).needsAttention,
    ).toBe(false);
  });
});

// @spec LAV-001 — Only actively executing agents are listed
describe("selectActiveConversations", () => {
  it("keeps running and waiting conversations and drops the rest", () => {
    const conversations = [
      conversation({ id: "run", execution_status: ExecutionStatus.RUNNING }),
      conversation({
        id: "wait",
        execution_status: ExecutionStatus.WAITING_FOR_CONFIRMATION,
      }),
      conversation({ id: "done", execution_status: ExecutionStatus.FINISHED }),
      conversation({ id: "idle", execution_status: ExecutionStatus.IDLE }),
      conversation({ id: "paused", execution_status: ExecutionStatus.PAUSED }),
      conversation({ id: "err", execution_status: ExecutionStatus.ERROR }),
      conversation({ id: "stuck", execution_status: ExecutionStatus.STUCK }),
    ];

    expect(
      selectActiveConversations(conversations).map((entry) => entry.id),
    ).toEqual(["run", "wait"]);
  });
});

// @spec LAV-003 — Subagent fan-out is derived from the event stream
describe("deriveSubagents", () => {
  it("opens a delegation per task action and keeps unresolved ones running", () => {
    const subagents = deriveSubagents([
      taskAction("c1", "explorer"),
      taskAction("c2", "coder"),
    ]);

    expect(subagents).toEqual([
      { id: "c1", name: "explorer", status: "running" },
      { id: "c2", name: "coder", status: "running" },
    ]);
  });

  it("closes a delegation as completed or errored from its observation", () => {
    // The observations reference the action event id, not the tool-call id.
    const subagents = deriveSubagents([
      taskAction("task-call-1", "explorer", "task-action-1"),
      taskObservation("task-action-1", false, "task-call-1"),
      taskAction("task-call-2", "coder", "task-action-2"),
      taskObservation("task-action-2", true, "task-call-2"),
    ]);

    expect(subagents).toEqual([
      { id: "task-call-1", name: "explorer", status: "completed" },
      { id: "task-call-2", name: "coder", status: "error" },
    ]);
  });

  it("ignores observations that do not pair with a task action", () => {
    expect(deriveSubagents([taskObservation("missing", false)])).toEqual([]);
  });
});

// @spec LAV-004 — Data is bounded and read-only
describe("mergeActivityTail", () => {
  it("carries an unresolved task action forward when the poll is gapless", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const previous = { events: [task, bashAction("c1")] };
    const next = [bashAction("c2"), bashAction("c3")];

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
    });

    expect(merged.events).toContain(task);
    expect(deriveSubagents(merged.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "running" },
    ]);
  });

  it("carries an unresolved action whose history scrolled out when the range is complete", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const previous = { events: [task], watermark: "2026-10-06T00:00:00Z" };
    const next = Array.from({ length: 60 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 1)).toISOString(),
    }));

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
    });

    expect(merged.events).toContain(task);
    expect(deriveSubagents(merged.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "running" },
    ]);
  });

  it("does not advance the watermark when the range is incomplete", () => {
    const previous = { events: [], watermark: "2026-10-06T00:00:00Z" };
    const newest = { ...bashAction("c9"), timestamp: "2026-10-06T00:00:30Z" };

    const merged = mergeActivityTail(previous, [newest], {
      canIncrementallyFetch: true,
      rangeComplete: false,
    });

    // The next poll must re-request the events between the old watermark and
    // `newest` that this partial page never read.
    expect(merged.watermark).toBe("2026-10-06T00:00:00Z");
  });

  it("commits the pending high watermark once a resumed range completes", () => {
    const previous = { events: [], watermark: "2026-10-06T00:00:00Z" };
    const newestPage = [
      { ...bashAction("c1"), timestamp: "2026-10-06T00:00:10Z" },
    ];

    const partial = mergeActivityTail(previous, newestPage, {
      canIncrementallyFetch: true,
      rangeComplete: false,
      resumePageId: "page-2",
    });
    // Unfinished range: keep the durable lower bound and remember the newest
    // timestamp for later.
    expect(partial.watermark).toBe("2026-10-06T00:00:00Z");
    expect(partial.pendingHighWatermark).toBe("2026-10-06T00:00:10Z");
    expect(partial.resumePageId).toBe("page-2");

    // A resumed page holds older events (or none at all). Committing only its
    // own maximum would leave the watermark at the start, so the backlog would
    // be re-fetched on every poll.
    const olderPage = [
      { ...bashAction("c2"), timestamp: "2026-10-06T00:00:05Z" },
    ];
    const completed = mergeActivityTail(partial, olderPage, {
      canIncrementallyFetch: true,
      rangeComplete: true,
    });
    expect(completed.watermark).toBe("2026-10-06T00:00:10Z");
    expect(completed.pendingHighWatermark).toBeUndefined();
    expect(completed.resumePageId).toBeUndefined();

    // An empty terminal page must still commit the remembered high watermark.
    const emptyTerminal = mergeActivityTail(partial, [], {
      canIncrementallyFetch: true,
      rangeComplete: true,
    });
    expect(emptyTerminal.watermark).toBe("2026-10-06T00:00:10Z");
  });

  it("drops a carried delegation when the range is incomplete", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const previous = { events: [task], watermark: "2026-10-06T00:00:00Z" };
    const next = Array.from({ length: 60 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 1)).toISOString(),
    }));

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
      rangeComplete: false,
    });

    // The action scrolled out of the bounded history and the incomplete range
    // cannot prove it is still running, so it must not be carried.
    expect(merged.events).toHaveLength(60);
    expect(deriveSubagents(merged.events)).toEqual([]);
    expect(merged.watermark).toBe("2026-10-06T00:00:00Z");
  });

  it("does not reopen a completed delegation whose observation is trimmed away", () => {
    // A task action and its observation are both near the start of the buffer.
    // Enough newer events arrive to displace the observation from the bounded
    // history; the action must not be carried as still-running just because
    // its resolution fell out of the window.
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const observation = taskObservation("task-action-1", false, "task-call-1");
    const previous = {
      events: [task, observation],
      watermark: "2026-10-06T00:00:01Z",
    };
    const next = Array.from({ length: 60 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 2)).toISOString(),
    }));

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
      rangeComplete: true,
    });

    expect(merged.events).toHaveLength(60);
    expect(deriveSubagents(merged.events)).toEqual([]);
  });

  it("stores a resolution during a poll before its observation leaves the window", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const observation = taskObservation("task-action-1", false, "task-call-1");

    // Poll 1: only the action has arrived.
    const first = mergeActivityTail(undefined, [task], {
      canIncrementallyFetch: true,
    });
    expect(deriveSubagents(first.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "running" },
    ]);

    // Poll 2: the observation arrives and its resolution is retained, not just
    // relied on being present in the bounded history.
    const second = mergeActivityTail(first, [observation], {
      canIncrementallyFetch: true,
    });
    expect(deriveSubagents(second.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "completed" },
    ]);
    expect(second.resolvedObservations).toContainEqual(observation);

    // Poll 3: enough newer events push the action and its observation out of
    // the bounded history. The delegation must never be reported as running.
    const third = mergeActivityTail(
      second,
      Array.from({ length: 60 }, (_, index) => ({
        ...bashAction(`c${index}`),
        timestamp: new Date(
          Date.UTC(2026, 9, 6, 0, 0, index + 5),
        ).toISOString(),
      })),
      { canIncrementallyFetch: true, rangeComplete: true },
    );
    expect(deriveSubagents(third.events)).toEqual([]);
  });

  it("keeps a carried completed delegation closed once its observation scrolls out", () => {
    // The resolution is retained separately from the bounded history, so the
    // carried action still reads as completed rather than reopening.
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const observation = taskObservation("task-action-1", false, "task-call-1");
    const previous = {
      events: [task],
      resolvedObservations: [observation],
      watermark: "2026-10-06T00:00:01Z",
    };
    const next = Array.from({ length: 60 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 2)).toISOString(),
    }));

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
      rangeComplete: true,
    });

    expect(merged.events).toContain(task);
    expect(deriveSubagents(merged.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "completed" },
    ]);
  });

  it("keeps a carried errored delegation errored once its observation scrolls out", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const observation = taskObservation("task-action-1", true, "task-call-1");
    const previous = {
      events: [task],
      resolvedObservations: [observation],
      watermark: "2026-10-06T00:00:01Z",
    };
    const next = Array.from({ length: 60 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 2)).toISOString(),
    }));

    const merged = mergeActivityTail(previous, next, {
      canIncrementallyFetch: true,
      rangeComplete: true,
    });

    expect(deriveSubagents(merged.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "error" },
    ]);
  });

  it("closes a carried delegation when its observation arrives later", () => {
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const previous = { events: [task] };
    const observation = taskObservation("task-action-1", false, "task-call-1");

    const merged = mergeActivityTail(previous, [observation], {
      canIncrementallyFetch: true,
    });

    // The action is retained so the observation can pair with it and close
    // the delegation as completed rather than dropping the row's subagent.
    expect(merged.events).toContain(task);
    expect(deriveSubagents(merged.events)).toEqual([
      { id: "task-call-1", name: "explorer", status: "completed" },
    ]);
  });

  it("drops a carried delegation when the window may contain gaps", () => {
    // The observation was produced and scrolled out between two polls. A
    // merge that cannot prove it saw every event must not keep claiming the
    // delegation is still running.
    const task = taskAction("task-call-1", "explorer", "task-action-1");
    const previous = { events: [task] };

    const merged = mergeActivityTail(previous, [bashAction("c9")], {
      canIncrementallyFetch: false,
    });

    expect(merged.events).toEqual([bashAction("c9")]);
    expect(deriveSubagents(merged.events)).toEqual([]);
  });

  it("records the newest timestamp as the next poll watermark", () => {
    const older = { ...bashAction("c1"), timestamp: "2026-10-06T00:00:00Z" };
    const newer = { ...bashAction("c2"), timestamp: "2026-10-06T00:00:09Z" };

    expect(mergeActivityTail(undefined, [older]).watermark).toBe(
      "2026-10-06T00:00:00Z",
    );
    expect(mergeActivityTail({ events: [older] }, [newer]).watermark).toBe(
      "2026-10-06T00:00:09Z",
    );
  });

  it("keeps the history bounded when a poll returns many events", () => {
    const events = Array.from({ length: 120 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index)).toISOString(),
    }));

    const merged = mergeActivityTail(undefined, events);

    expect(merged.events).toHaveLength(60);
    expect(merged.events.at(-1)).toEqual(events.at(-1));
  });

  it("remembers a backend without timestamp filters", () => {
    const merged = mergeActivityTail(undefined, [bashAction("c1")], {
      supportsTimestampFilter: false,
    });

    expect(merged.supportsTimestampFilter).toBe(false);
  });
});

// @spec LAV-002 — A row conveys status, current step, and spend
describe("pickLatestActivity", () => {
  it("prefers the newest action over an earlier assistant message", () => {
    const descriptor = pickLatestActivity([
      assistantMessage("Let me check"),
      bashAction("c1"),
    ]);

    expect(descriptor).toEqual({
      kind: "translation",
      key: "ACTION_MESSAGE$RUN",
      values: { command: "npm test" },
    });
  });

  it("prefers a newer assistant message over an earlier action", () => {
    const descriptor = pickLatestActivity([
      bashAction("c1"),
      assistantMessage("Tests passed; inspecting the diff."),
    ]);

    expect(descriptor).toEqual({
      kind: "text",
      text: "Tests passed; inspecting the diff.",
    });
  });

  it("falls back to the last assistant message when there is no action", () => {
    expect(pickLatestActivity([assistantMessage("All done")])).toEqual({
      kind: "text",
      text: "All done",
    });
  });

  it("returns null when there is nothing to show", () => {
    expect(pickLatestActivity([])).toBeNull();
  });
});

// @spec LAV-002 — A row conveys status, current step, and spend
describe("getActivityMetrics", () => {
  it("sums prompt and completion tokens and keeps the cost", () => {
    expect(
      getActivityMetrics({
        accumulated_cost: 0.25,
        max_budget_per_task: null,
        accumulated_token_usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          context_window: 0,
          per_turn_token: 0,
        },
      }),
    ).toEqual({ cost: 0.25, totalTokens: 120 });
  });

  it("returns nulls when metrics are missing", () => {
    expect(getActivityMetrics(null)).toEqual({
      cost: null,
      totalTokens: null,
    });
  });
});
