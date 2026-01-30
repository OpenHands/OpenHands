import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import type { V1AppConversationStartTask } from "#/api/conversation-service/v1-conversation-service.types";

// Mock the underlying service
vi.mock("#/api/conversation-service/v1-conversation-service.api", () => ({
  default: {
    getStartTask: vi.fn(),
  },
}));

describe("useSubConversationTaskPolling", () => {
  let queryClient: QueryClient;

  const createWrapper = () => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

  const createMockTask = (
    status: V1AppConversationStartTask["status"],
    appConversationId: string | null = null,
  ): V1AppConversationStartTask => ({
    id: "task-123",
    created_by_user_id: "user-1",
    status,
    detail: null,
    app_conversation_id: appConversationId,
    sandbox_id: null,
    agent_server_url: null,
    request: {},
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    queryClient?.clear();
  });

  it("should return task status when task is READY", async () => {
    // Arrange
    const mockTask = createMockTask("READY", "sub-conversation-123");
    vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(mockTask);

    // Act
    const { result } = renderHook(
      () =>
        useSubConversationTaskPolling("task-123", "parent-conversation-456"),
      { wrapper: createWrapper() },
    );

    // Assert
    await waitFor(() => {
      expect(result.current.taskStatus).toBe("READY");
    });
    expect(result.current.subConversationId).toBe("sub-conversation-123");
    expect(V1ConversationService.getStartTask).toHaveBeenCalledWith("task-123");
  });

  it("should not poll when taskId is null", async () => {
    // Arrange
    vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(null);

    // Act
    const { result } = renderHook(
      () => useSubConversationTaskPolling(null, "parent-conversation-456"),
      { wrapper: createWrapper() },
    );

    // Assert - wait a bit to ensure no calls are made
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(V1ConversationService.getStartTask).not.toHaveBeenCalled();
    expect(result.current.taskStatus).toBeUndefined();
  });

  it("should not poll when parentConversationId is null", async () => {
    // Arrange
    vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(null);

    // Act
    const { result } = renderHook(
      () => useSubConversationTaskPolling("task-123", null),
      { wrapper: createWrapper() },
    );

    // Assert - wait a bit to ensure no calls are made
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    expect(V1ConversationService.getStartTask).not.toHaveBeenCalled();
    expect(result.current.taskStatus).toBeUndefined();
  });
});
