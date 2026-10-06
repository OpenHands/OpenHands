import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { useActivityEventTails } from "#/hooks/query/use-activity-event-tails";
import { CONVERSATION_QUERY_KEYS } from "#/hooks/query/query-keys";

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

    // Same conversation id and runtime URL, new session. When the next poll
    // runs it must ignore the previous session's buffer, so the unresolved
    // delegation is not carried into the new tail.
    searchEvents.mockResolvedValueOnce({ items: [bashAction("bash-1")] });
    rerender({ key: "new" });
    await refetchTails(client);

    await waitFor(() =>
      expect(result.current[0]).toEqual([bashAction("bash-1")]),
    );
    expect(result.current[0]).not.toContainEqual(
      expect.objectContaining({ id: "old-action" }),
    );

    // The session key is used for the request but never enters the key.
    const queryKeys = client
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
    expect(queryKeys.some((key) => key.includes("old"))).toBe(false);
    expect(queryKeys.some((key) => key.includes("new"))).toBe(false);
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
});
