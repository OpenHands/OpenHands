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

const taskAction = (eventId: string): OpenHandsEvent =>
  ({
    id: eventId,
    timestamp: "2026-10-06T00:00:00Z",
    source: "agent",
    action: { kind: "TaskAction", subagent_type: "explorer" },
    tool_name: "task",
    tool_call_id: `call-${eventId}`,
  }) as unknown as OpenHandsEvent;

const bashAction = (eventId: string): OpenHandsEvent =>
  ({
    id: eventId,
    timestamp: "2026-10-06T00:00:01Z",
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

beforeEach(() => {
  vi.clearAllMocks();
  backendMock.current = {
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  };
});

describe("useActivityEventTails", () => {
  it("carries an unresolved task action forward across refetches", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const task = taskAction("task-action-1");
    searchEvents.mockResolvedValueOnce({ items: [task] });

    const { result } = renderHook(
      () => useActivityEventTails([conversation()]),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current[0]).toEqual([task]));

    // The next poll no longer contains the still-running task action.
    searchEvents.mockResolvedValueOnce({ items: [bashAction("bash-1")] });
    await client.refetchQueries({
      queryKey: [...CONVERSATION_QUERY_KEYS.activityTail],
    });

    await waitFor(() =>
      expect(result.current[0]).toEqual([task, bashAction("bash-1")]),
    );
  });

  it("starts a fresh tail when the runtime URL changes", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
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

    // Re-provisioned on a new sandbox: same id, new URL.
    const fresh = taskObservation("new-action");
    searchEvents.mockResolvedValueOnce({ items: [fresh] });
    rerender({ url: "http://runtime/new" });

    await waitFor(() => expect(result.current[0]).toEqual([fresh]));
    expect(result.current[0]).not.toContainEqual(
      expect.objectContaining({ id: "old-action" }),
    );
  });
});
