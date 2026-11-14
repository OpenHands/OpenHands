import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useConversationStore } from "#/stores/conversation-store";
import { setConversationState } from "#/utils/conversation-local-storage";

vi.mock("#/api/conversation-service/agent-server-conversation-service.api", () => ({
  default: {
    getStartTask: vi.fn(),
  },
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: vi.fn(),
}));

vi.mock("#/utils/conversation-local-storage", () => ({
  setConversationState: vi.fn(),
}));

describe("useSubConversationTaskPolling", () => {
  let queryClient: QueryClient;
  const setSubConversationTaskId = vi.fn();

  const createWrapper = () => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return function ({ children }: { children: React.ReactNode }) {
      return (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      );
    };
  };

  const createMockTask = (
    status: AppConversationStartTask["status"],
    appConversationId: string | null = null,
  ): AppConversationStartTask => ({
    id: "task-123",
    created_by_user_id: "user-1",
    status,
    detail: null,
    app_conversation_id: appConversationId,
    agent_server_url: null,
    request: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useConversationStore).mockImplementation((selector: unknown) => {
      const state = { setSubConversationTaskId };
      if (typeof selector === "function") {
        return (selector as (s: typeof state) => unknown)(state);
      }
      return state;
    });
  });

  afterEach(() => {
    queryClient?.clear();
  });

  it("should return task status when task is READY", async () => {
    const mockTask = createMockTask("READY", "sub-conversation-123");
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      mockTask,
    );

    const { result } = renderHook(
      () =>
        useSubConversationTaskPolling("task-123", "parent-conversation-456"),
      { wrapper: createWrapper() },
    );

    await waitFor(() => {
      expect(result.current.taskStatus).toBe("READY");
    });
    expect(result.current.subConversationId).toBe("sub-conversation-123");
    expect(AgentServerConversationService.getStartTask).toHaveBeenCalledWith(
      "task-123",
    );
    expect(setSubConversationTaskId).not.toHaveBeenCalled();
    expect(setConversationState).not.toHaveBeenCalled();
  });

  it("clears failed Cloud planner ownership so Plan can retry (#19380)", async () => {
    const mockTask = createMockTask("ERROR", null);
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      mockTask,
    );

    const { result } = renderHook(
      () =>
        useSubConversationTaskPolling(
          "research-planner-start",
          "parent-conversation-456",
        ),
      { wrapper: createWrapper() },
    );

    await waitFor(() => {
      expect(result.current.taskStatus).toBe("ERROR");
    });
    expect(setSubConversationTaskId).toHaveBeenCalledWith(null);
    expect(setConversationState).toHaveBeenCalledWith(
      "parent-conversation-456",
      { subConversationTaskId: null },
    );
  });

  it("does not clear ownership when ERROR still produced a conversation id", async () => {
    const mockTask = createMockTask("ERROR", "sub-conversation-123");
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      mockTask,
    );

    const { result } = renderHook(
      () =>
        useSubConversationTaskPolling("task-123", "parent-conversation-456"),
      { wrapper: createWrapper() },
    );

    await waitFor(() => {
      expect(result.current.taskStatus).toBe("ERROR");
    });
    expect(setSubConversationTaskId).not.toHaveBeenCalled();
    expect(setConversationState).not.toHaveBeenCalled();
  });

  it("should not poll when taskId is null", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      null,
    );

    const { result } = renderHook(
      () => useSubConversationTaskPolling(null, "parent-conversation-456"),
      { wrapper: createWrapper() },
    );

    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(AgentServerConversationService.getStartTask).not.toHaveBeenCalled();
    expect(result.current.taskStatus).toBeUndefined();
  });

  it("should not poll when parentConversationId is null", async () => {
    vi.mocked(AgentServerConversationService.getStartTask).mockResolvedValue(
      null,
    );

    const { result } = renderHook(
      () => useSubConversationTaskPolling("task-123", null),
      { wrapper: createWrapper() },
    );

    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(AgentServerConversationService.getStartTask).not.toHaveBeenCalled();
    expect(result.current.taskStatus).toBeUndefined();
  });
});
