import React from "react";
import { HttpError } from "@openhands/typescript-client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { NavigationProvider } from "#/context/navigation-context";
import {
  useTaskPolling,
  useTaskPollingController,
} from "#/hooks/query/use-task-polling";
import { trackCloudConversationReady } from "#/services/cloud-funnel-analytics";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import {
  consumePendingTaskDraft,
  getConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";
import { resetPendingTaskMessageLinkState } from "#/utils/pending-task-message-link";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      getStartTask: vi.fn(),
    },
  }),
);
vi.mock("#/services/cloud-funnel-analytics", () => ({
  trackCloudConversationReady: vi.fn(),
}));

const readyTask: AppConversationStartTask = {
  id: "123",
  created_by_user_id: "user-1",
  status: "READY",
  detail: null,
  app_conversation_id: "conversation-1",
  agent_server_url: null,
  request: {},
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

describe("useTaskPolling", () => {
  let queryClient: QueryClient;
  const navigate = vi.fn();

  const createWrapper = () => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, retryDelay: 1 } },
    });

    return function Wrapper({ children }: { children: React.ReactNode }) {
      return (
        <QueryClientProvider client={queryClient}>
          <NavigationProvider
            value={{
              currentPath: "/conversations/task-123",
              conversationId: "task-123",
              isNavigating: false,
              navigate,
            }}
          >
            {children}
          </NavigationProvider>
        </QueryClientProvider>
      );
    };
  };

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    resetPendingTaskMessageLinkState();
    useOptimisticUserMessageStore.getState().clearPendingMessages();
  });

  afterEach(() => {
    queryClient?.clear();
    vi.useRealTimers();
    localStorage.clear();
    sessionStorage.clear();
  });

  it("moves pending task drafts onto the real conversation before redirecting", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      readyTask,
    );
    setPendingTaskDraft("123", "Create this automation draft");

    const { result } = renderHook(() => useTaskPollingController(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.taskStatus).toBe("READY"));
    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(
        "/conversations/conversation-1?backend=default-local",
        {
          replace: true,
        },
      );
    });

    expect(getConversationState("conversation-1").draftMessage).toBe(
      "Create this automation draft",
    );
    expect(consumePendingTaskDraft("123")).toBeNull();
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledWith(
      "123",
    );
  });

  it("stores cloud task plugin coordinates on the real conversation", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue({
      ...readyTask,
      request: {
        selected_repository: "OpenHands/agent-canvas",
        selected_branch: "main",
        git_provider: "github",
        plugins: [
          {
            source: "github:OpenHands/extensions",
            ref: "v1",
            repo_path: "plugins/weather",
            parameters: { apiKey: "secret" },
          },
        ],
      },
    });

    renderHook(() => useTaskPollingController(), { wrapper: createWrapper() });

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(
        "/conversations/conversation-1?backend=default-local",
        {
          replace: true,
        },
      );
    });
    expect(getStoredConversationMetadata("conversation-1")).toEqual({
      selected_repository: "OpenHands/agent-canvas",
      selected_branch: "main",
      git_provider: "github",
      plugins: [
        {
          source: "github:OpenHands/extensions",
          ref: "v1",
          repo_path: "plugins/weather",
        },
      ],
    });
  });

  it("reassigns optimistic pending messages on the real conversation route", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      readyTask,
    );
    useOptimisticUserMessageStore.getState().enqueuePendingMessage({
      conversationId: "task-123",
      text: "hello from home",
    });

    const createWrapperForConversation = (conversationId: string) => {
      queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false, retryDelay: 1 } },
      });

      return function Wrapper({ children }: { children: React.ReactNode }) {
        return (
          <QueryClientProvider client={queryClient}>
            <NavigationProvider
              value={{
                currentPath: `/conversations/${conversationId}`,
                conversationId,
                isNavigating: false,
                navigate,
              }}
            >
              {children}
            </NavigationProvider>
          </QueryClientProvider>
        );
      };
    };

    renderHook(() => useTaskPollingController(), {
      wrapper: createWrapperForConversation("task-123"),
    });

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(
        "/conversations/conversation-1?backend=default-local",
        {
          replace: true,
        },
      );
    });

    renderHook(() => useTaskPollingController(), {
      wrapper: createWrapperForConversation("conversation-1"),
    });

    await waitFor(() => {
      const pending = useOptimisticUserMessageStore.getState().pendingMessages;
      expect(pending).toHaveLength(1);
      expect(pending[0].conversationId).toBe("conversation-1");
      expect(pending[0].text).toBe("hello from home");
    });
  });

  it("marks the sidebar conversation list stale once the task is ready", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      readyTask,
    );
    const wrapper = createWrapper();
    // Same key shapes `usePaginatedConversations` and `useStartTasks` read.
    const conversationsKey = [
      "user",
      "conversations",
      "paginated",
      20,
      "default-local",
      null,
    ];
    const startTasksKey = ["start-tasks", "search", 10];
    queryClient.setQueryData(conversationsKey, { pages: [], pageParams: [] });
    queryClient.setQueryData(startTasksKey, []);

    renderHook(() => useTaskPollingController(), { wrapper });

    await waitFor(() => expect(navigate).toHaveBeenCalledOnce());
    expect(queryClient.getQueryState(conversationsKey)?.isInvalidated).toBe(
      true,
    );
    expect(queryClient.getQueryState(startTasksKey)?.isInvalidated).toBe(true);
  });

  it("handles a ready task once when multiple components consume polling state", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      readyTask,
    );

    const { result } = renderHook(
      () => {
        const controller = useTaskPollingController();
        useTaskPolling();
        useTaskPolling();
        return controller;
      },
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.taskStatus).toBe("READY"));
    await waitFor(() => expect(navigate).toHaveBeenCalledTimes(1));

    expect(trackCloudConversationReady).toHaveBeenCalledOnce();
    expect(trackCloudConversationReady).toHaveBeenCalledWith(
      "123",
      "conversation-1",
    );
  });
  it("recovers the first task read after a transient failure and navigates once", async () => {
    vi.mocked(AgentServerConversationService.getStartTask)
      .mockRejectedValueOnce(new HttpError(503, "Unavailable", null))
      .mockResolvedValue(readyTask);
    renderHook(() => useTaskPollingController(), { wrapper: createWrapper() });
    await waitFor(() => expect(navigate).toHaveBeenCalledOnce());
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      2,
    );
  });

  it("stops polling after a terminal request error", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockRejectedValue(
      new HttpError(404, "Not found", null),
    );
    const { result } = renderHook(() => useTaskPolling(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.taskError).toBeTruthy());
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      1,
    );
    expect(navigate).not.toHaveBeenCalled();
  });
  it("discovers a ready session within a second and stops polling", async () => {
    vi.useFakeTimers();
    vi.mocked(AgentServerConversationService.getStartTask)
      .mockResolvedValueOnce({
        ...readyTask,
        status: "WORKING",
        app_conversation_id: null,
      })
      .mockResolvedValue(readyTask);
    const { result } = renderHook(() => useTaskPolling(), {
      wrapper: createWrapper(),
    });
    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(result.current.taskStatus).toBe("READY");
    const calls = vi.mocked(AgentServerConversationService.getStartTask).mock
      .calls.length;
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      calls,
    );
  });

  it("backs off long starts and releases the polling timer on unmount", async () => {
    vi.useFakeTimers();
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue({
      ...readyTask,
      status: "WORKING",
      app_conversation_id: null,
    });
    const { unmount } = renderHook(() => useTaskPolling(), {
      wrapper: createWrapper(),
    });
    await act(() => vi.advanceTimersByTimeAsync(31000));
    const calls = vi.mocked(AgentServerConversationService.getStartTask).mock
      .calls.length;
    await act(() => vi.advanceTimersByTimeAsync(3000));
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      calls + 1,
    );
    unmount();
    await act(() => vi.advanceTimersByTimeAsync(10000));
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledTimes(
      calls + 1,
    );
  });
});
