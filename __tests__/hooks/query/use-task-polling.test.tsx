import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
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
      defaultOptions: { queries: { retry: false } },
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
    resetPendingTaskMessageLinkState();
    useOptimisticUserMessageStore.setState({ pendingMessages: [] });
  });

  afterEach(() => {
    queryClient?.clear();
    localStorage.clear();
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
      expect(navigate).toHaveBeenCalledWith("/conversations/conversation-1", {
        replace: true,
      });
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
      expect(navigate).toHaveBeenCalledWith("/conversations/conversation-1", {
        replace: true,
      });
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
        defaultOptions: { queries: { retry: false } },
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
      expect(navigate).toHaveBeenCalledWith("/conversations/conversation-1", {
        replace: true,
      });
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
});
