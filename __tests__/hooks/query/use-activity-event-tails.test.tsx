import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  useActivityEventTails,
  __resetActivitySessionGenerationsForTests,
} from "#/hooks/query/use-activity-event-tails";
import { CONVERSATION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { deriveSubagents } from "#/components/features/activity/activity-view-model";

const backendMock = vi.hoisted(() => ({
  current: {
    backend: { id: "local-1", kind: "local" as "local" | "cloud" },
    orgId: null as string | null,
  },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => backendMock.current,
}));

const searchEvents = vi.hoisted(() => vi.fn());
vi.mock("#/api/event-service/event-service.api", () => ({
  default: {
    searchEvents: (...args: unknown[]) => searchEvents(...args),
  },
}));

const conversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation =>
  ({
    id: "conv-1",
    conversation_url: "http://runtime/conv-1",
    session_api_key: "key-1",
    ...overrides,
  }) as AppConversation;

const taskAction = (
  eventId: string,
  timestamp = "2026-10-06T00:00:00Z",
): OpenHandsEvent =>
  ({
    id: eventId,
    timestamp,
    source: "agent",
    action: { kind: "TaskAction", subagent_type: "explorer" },
    tool_name: "task",
    tool_call_id: `call-${eventId}`,
  }) as unknown as OpenHandsEvent;

const bashAction = (
  eventId: string,
  timestamp = "2026-10-06T00:00:01Z",
): OpenHandsEvent =>
  ({
    id: eventId,
    timestamp,
    source: "agent",
    action: { kind: "ExecuteBashAction", command: "ls" },
    tool_name: "execute_bash",
    tool_call_id: `call-${eventId}`,
  }) as unknown as OpenHandsEvent;

const taskObservation = (actionEventId: string): OpenHandsEvent =>
  ({
    id: `obs-${actionEventId}`,
    timestamp: "2026-10-06T00:00:02Z",
    source: "environment",
    action_id: actionEventId,
    tool_name: "task",
    tool_call_id: `call-${actionEventId}`,
    observation: { kind: "TaskObservation", is_error: false },
  }) as unknown as OpenHandsEvent;

function makeWrapper(client: QueryClient) {
  return function wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

function refetchTails(client: QueryClient) {
  return client.refetchQueries({
    queryKey: [...CONVERSATION_QUERY_KEYS.activityTail],
  });
}

beforeEach(() => {
  vi.resetAllMocks();
  __resetActivitySessionGenerationsForTests();
  backendMock.current = {
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  };
});

describe("useActivityEventTails", () => {
  it("requests everything since the last poll so no observation is missed", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));
    // The first poll has no watermark yet, so it must not send a filter.
    expect(searchEvents.mock.calls[0][3]).not.toHaveProperty("timestampGte");

    // The next poll asks for events at or after the newest one seen.
    searchEvents.mockResolvedValueOnce({ items: [] });
    await refetchTails(client);

    await waitFor(() => expect(searchEvents).toHaveBeenCalledTimes(2));
    expect(searchEvents.mock.calls[1][3]).toMatchObject({
      timestampGte: "2026-10-06T00:00:00Z",
    });
  });

  it("closes a delegation whose observation arrives in a later poll", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    const observation = taskObservation("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [observation] });
    await refetchTails(client);

    await waitFor(() => expect(result.current[0]).toContainEqual(observation));
    expect(result.current[0]).toContainEqual(task);
  });

  it("starts a fresh tail when the runtime URL changes", async () => {
    const client = newClient();
    searchEvents.mockResolvedValueOnce({ items: [taskAction("old-action")] });

    const { result, rerender } = renderHook(
      ({ url }: { url: string }) =>
        useActivityEventTails([conversation({ conversation_url: url })]),
      {
        initialProps: { url: "http://runtime/old" },
        wrapper: makeWrapper(client),
      },
    );

    await waitFor(() => expect(result.current[0]).toHaveLength(1));

    const fresh = taskObservation("new-action");
    searchEvents.mockResolvedValueOnce({ items: [fresh] });
    rerender({ url: "http://runtime/new" });

    await waitFor(() => expect(result.current[0]).toEqual([fresh]));
    expect(result.current[0]).not.toContainEqual(
      expect.objectContaining({ id: "old-action" }),
    );
  });

  it("does not carry a delegation across a rotated session key", async () => {
    const client = newClient();
    searchEvents.mockResolvedValueOnce({ items: [taskAction("old-action")] });

    const { result, rerender } = renderHook(
      ({ key }: { key: string }) =>
        useActivityEventTails([conversation({ session_api_key: key })]),
      {
        initialProps: { key: "old" },
        wrapper: makeWrapper(client),
      },
    );

    await waitFor(() => expect(result.current[0]).toHaveLength(1));

    // Same conversation id and runtime URL, new session. The rotation must
    // clear the cached tail immediately, before the next periodic refetch, so
    // the row does not keep showing the previous session's activity.
    searchEvents.mockResolvedValueOnce({ items: [bashAction("bash-1")] });
    rerender({ key: "new" });

    await waitFor(() =>
      expect(result.current[0]).toEqual([bashAction("bash-1")]),
    );
    expect(result.current[0]).not.toContainEqual(
      expect.objectContaining({ id: "old-action" }),
    );

    // The session key is used for the request but never enters the key or the
    // cached value.
    const queryKeys = client
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
    expect(queryKeys.some((key) => key.includes("old"))).toBe(false);
    expect(queryKeys.some((key) => key.includes("new"))).toBe(false);

    const cachedValues = JSON.stringify(
      client
        .getQueryCache()
        .getAll()
        .map((query) => query.state.data),
    );
    expect(cachedValues).not.toContain("key-1");
    expect(cachedValues).not.toContain("old");
    expect(cachedValues).not.toContain("new");
  });

  it("paginates the range so a burst larger than one page still closes a delegation", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    // The observation and a full page of newer events arrive before the next
    // poll. The observation is on the second page, so the poll must follow
    // `next_page_id` rather than only reading the newest 30 events.
    const newer = Array.from({ length: 30 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 1)).toISOString(),
    }));
    const observation = taskObservation("task-action-1");
    searchEvents
      .mockResolvedValueOnce({ items: newer, next_page_id: "page-2" })
      .mockResolvedValueOnce({ items: [observation], next_page_id: null });

    await refetchTails(client);

    await waitFor(() => expect(result.current[0]).toContainEqual(observation));
    expect(searchEvents).toHaveBeenCalledTimes(3);
    expect(searchEvents.mock.calls[1][3]).toMatchObject({
      timestampGte: "2026-10-06T00:00:00Z",
    });
    expect(searchEvents.mock.calls[2][3]).toMatchObject({
      pageId: "page-2",
      timestampGte: "2026-10-06T00:00:00Z",
    });
    expect(deriveSubagents(result.current[0] ?? [])).toEqual([
      { id: "call-task-action-1", name: "explorer", status: "completed" },
    ]);
  });

  it("falls back to an unfiltered tail when the backend rejects the filter", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    // A filter that is not supported fails once, then the plain tail is read.
    searchEvents.mockRejectedValueOnce(new Error("unsupported filter"));
    searchEvents.mockResolvedValueOnce({ items: [bashAction("bash-1")] });
    await refetchTails(client);

    await waitFor(() =>
      expect(result.current[0]).toEqual([bashAction("bash-1")]),
    );

    // The delegation is dropped rather than reported as running forever.
    expect(result.current[0]).not.toContainEqual(task);
  });

  it("resumes an unfinished backlog instead of re-reading the newest pages", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    // A backlog larger than the page bound: every poll returns a next page so
    // the range never completes within one poll.
    const backlogPage = (offset: number) => ({
      items: Array.from({ length: 30 }, (_, index) => ({
        ...bashAction(`c${offset + index}`),
        timestamp: new Date(
          Date.UTC(2026, 9, 6, 0, 0, offset + index + 1),
        ).toISOString(),
      })),
      next_page_id: `page-${offset + 30}`,
    });

    for (let page = 0; page < 20; page += 1) {
      searchEvents.mockResolvedValueOnce(backlogPage(page * 30));
    }
    await refetchTails(client);
    await waitFor(() => expect(searchEvents).toHaveBeenCalledTimes(21));

    // The last read page in the capped poll is the 20th backlog page; its
    // cursor (page-600) is what the next poll must resume from.
    const lastBacklogCall = searchEvents.mock.calls[20][3] as {
      pageId?: string;
    };
    expect(lastBacklogCall.pageId).toBe("page-570");

    searchEvents.mockResolvedValueOnce({ items: [], next_page_id: null });
    await refetchTails(client);
    await waitFor(() => expect(searchEvents).toHaveBeenCalledTimes(22));

    // The next poll continues from the stored cursor rather than re-reading
    // the same newest pages.
    const resumeCall = searchEvents.mock.calls[21][3] as { pageId?: string };
    expect(resumeCall.pageId).toBe("page-600");
  });

  it("does not advance past events when a later cloud page fails", async () => {
    const client = newClient();
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    // The first page succeeds and points at a second page holding the
    // observation; that page fails. The failed range must not advance the
    // watermark, so the observation is not skipped permanently.
    const newer = Array.from({ length: 30 }, (_, index) => ({
      ...bashAction(`c${index}`),
      timestamp: new Date(Date.UTC(2026, 9, 6, 0, 0, index + 1)).toISOString(),
    }));
    searchEvents
      .mockResolvedValueOnce({ items: newer, next_page_id: "page-2" })
      .mockRejectedValueOnce(new Error("page 2 failed"));

    await refetchTails(client);
    await waitFor(() => expect(searchEvents).toHaveBeenCalledTimes(3));

    // The next poll retries the failed page with the original watermark.
    searchEvents.mockResolvedValueOnce({
      items: [taskObservation("task-action-1")],
      next_page_id: null,
    });
    await refetchTails(client);
    await waitFor(() =>
      expect(deriveSubagents(result.current[0] ?? [])).toEqual([
        { id: "call-task-action-1", name: "explorer", status: "completed" },
      ]),
    );
    expect(searchEvents.mock.calls.at(-1)?.[3]).toMatchObject({
      pageId: "page-2",
      timestampGte: "2026-10-06T00:00:00Z",
    });
  });

  it("clears a stale tail when a conversation returns with a rotated key after remounting", async () => {
    const client = newClient();
    searchEvents.mockResolvedValueOnce({ items: [taskAction("old-action")] });

    const mounted = renderHook(
      () => useActivityEventTails([conversation({ session_api_key: "old" })]),
      { wrapper: makeWrapper(client) },
    );
    await waitFor(() => expect(mounted.result.current[0]).toHaveLength(1));
    mounted.unmount();

    // The view remounts against the same query cache, but the session key
    // rotated while the conversation was inactive. The cached tail must be
    // cleared even though the query key is unchanged.
    searchEvents.mockResolvedValueOnce({ items: [bashAction("new-action")] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation({ session_api_key: "new" })]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() =>
      expect(result.current[0]).toEqual([bashAction("new-action")]),
    );
    expect(result.current[0]).not.toContainEqual(
      expect.objectContaining({ id: "old-action" }),
    );
  });
});
