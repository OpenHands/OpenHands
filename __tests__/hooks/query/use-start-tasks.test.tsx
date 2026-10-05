import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useStartTasks } from "#/hooks/query/use-start-tasks";
import {
  trackStartTask,
  useInFlightStartTasksStore,
} from "#/stores/in-flight-start-tasks-store";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      getStartTasks: vi.fn(),
    },
  }),
);

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "cloud-1", kind: "cloud" as const },
    orgId: null,
  }),
}));

const makeTask = (
  id: string,
  status: AppConversationStartTask["status"],
): AppConversationStartTask => ({
  id,
  created_by_user_id: "user-1",
  status,
  detail: null,
  app_conversation_id: status === "READY" ? `conversation-${id}` : null,
  agent_server_url: null,
  request: {},
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
});

const getStartTasksMock = vi.mocked(
  AgentServerConversationService.getStartTasks,
);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
};

const trackedIds = () =>
  useInFlightStartTasksStore.getState().taskIdsByBackendId["cloud-1"] ?? [];

describe("useStartTasks", () => {
  beforeEach(() => {
    useInFlightStartTasksStore.setState({ taskIdsByBackendId: {} });
  });

  afterEach(() => {
    getStartTasksMock.mockReset();
  });

  it("returns no tasks and skips the request when nothing is in flight", async () => {
    const { result } = renderHook(() => useStartTasks(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isFetching).toBe(false));
    expect(result.current.data).toBeUndefined();
    expect(getStartTasksMock).not.toHaveBeenCalled();
  });

  it("fetches the tracked task ids and exposes in-progress tasks", async () => {
    trackStartTask("cloud-1", "task-1");
    getStartTasksMock.mockResolvedValue([
      makeTask("task-1", "WAITING_FOR_SANDBOX"),
    ]);

    const { result } = renderHook(() => useStartTasks(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.data).toHaveLength(1));
    expect(getStartTasksMock).toHaveBeenCalledWith(["task-1"]);
    expect(result.current.data?.[0].id).toBe("task-1");
  });

  it("drops settled tasks from the list and stops tracking them", async () => {
    trackStartTask("cloud-1", "ready-task");
    trackStartTask("cloud-1", "working-task");
    getStartTasksMock.mockResolvedValue([
      makeTask("working-task", "PREPARING_REPOSITORY"),
      makeTask("ready-task", "READY"),
    ]);

    const { result } = renderHook(() => useStartTasks(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.data).toHaveLength(1));
    expect(result.current.data?.[0].id).toBe("working-task");
    await waitFor(() => expect(trackedIds()).toEqual(["working-task"]));
  });

  it("stops tracking ids the backend no longer knows", async () => {
    trackStartTask("cloud-1", "forgotten-task");
    getStartTasksMock.mockResolvedValue([]);

    renderHook(() => useStartTasks(), { wrapper: createWrapper() });

    await waitFor(() => expect(trackedIds()).toEqual([]));
  });

  it("keeps tracking ids when the batch request fails", async () => {
    trackStartTask("cloud-1", "task-1");
    getStartTasksMock.mockRejectedValue(new Error("network down"));

    const { result } = renderHook(() => useStartTasks(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(trackedIds()).toEqual(["task-1"]);
  });
});
