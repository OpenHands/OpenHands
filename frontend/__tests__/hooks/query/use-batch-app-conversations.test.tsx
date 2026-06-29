import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useBatchAppConversations } from "#/hooks/query/use-batch-app-conversations";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useBatchAppConversations", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not call the app-conversations endpoint for task IDs", () => {
    // Arrange: a task ID (format "task-{uuid}") is not a valid conversation UUID
    const batchGetAppConversations = vi
      .spyOn(V1ConversationService, "batchGetAppConversations")
      .mockResolvedValue([]);

    // Act
    renderHook(
      () => useBatchAppConversations(["task-949349dbeae346e895e25679b16f8193"]),
      { wrapper: createWrapper() },
    );

    // Assert: the endpoint is never hit, so the backend never returns a 400
    expect(batchGetAppConversations).not.toHaveBeenCalled();
  });

  it("calls the app-conversations endpoint for valid conversation IDs", async () => {
    // Arrange
    const conversationId = "5c99c669-1151-4413-a75d-1fa4f2cc5355";
    const batchGetAppConversations = vi
      .spyOn(V1ConversationService, "batchGetAppConversations")
      .mockResolvedValue([]);

    // Act
    renderHook(() => useBatchAppConversations([conversationId]), {
      wrapper: createWrapper(),
    });

    // Assert
    await waitFor(() =>
      expect(batchGetAppConversations).toHaveBeenCalledWith([conversationId]),
    );
  });
});
