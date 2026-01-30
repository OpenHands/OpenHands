import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { setConversationState } from "#/utils/conversation-local-storage";
import { useConversationStore } from "#/stores/conversation-store";
import type { V1AppConversationStartTask } from "#/api/conversation-service/v1-conversation-service.types";

// Mock dependencies
vi.mock("#/api/conversation-service/v1-conversation-service.api");
vi.mock("#/utils/conversation-local-storage");
vi.mock("#/stores/conversation-store");

const mockSetSubConversationTaskId = vi.fn();
const mockInvalidateQueries = vi.fn();

// Helper function to create properly typed mock return values
function asMockReturnValue<T>(value: Partial<T>): T {
  return value as T;
}

function makeTask(
  status: V1AppConversationStartTask["status"],
  appConversationId: string | null = null,
): V1AppConversationStartTask {
  return {
    id: "task-123",
    created_by_user_id: "user-123",
    status,
    detail: null,
    app_conversation_id: appConversationId,
    sandbox_id: null,
    agent_server_url: null,
    request: {
      agent_type: "plan",
      parent_conversation_id: "parent-conv-123",
    },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
}

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
}

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = createQueryClient();
  queryClient.invalidateQueries = mockInvalidateQueries;
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useSubConversationTaskPolling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useConversationStore).mockReturnValue(
      asMockReturnValue<ReturnType<typeof useConversationStore>>({
        setSubConversationTaskId: mockSetSubConversationTaskId,
      }),
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("when task status is READY", () => {
    it("clears subConversationTaskId from localStorage and store when task completes successfully", async () => {
      const parentConversationId = "parent-conv-123";
      const taskId = "task-123";
      const appConversationId = "sub-conv-456";

      const readyTask = makeTask("READY", appConversationId);
      vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(
        readyTask,
      );

      renderHook(
        () => useSubConversationTaskPolling(taskId, parentConversationId),
        { wrapper },
      );

      await waitFor(() => {
        expect(setConversationState).toHaveBeenCalledWith(
          parentConversationId,
          { subConversationTaskId: null },
        );
      });

      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(null);
      expect(mockInvalidateQueries).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", parentConversationId],
      });
    });

    it("invalidates parent conversation cache when task is READY", async () => {
      const parentConversationId = "parent-conv-123";
      const taskId = "task-123";
      const appConversationId = "sub-conv-456";

      const readyTask = makeTask("READY", appConversationId);
      vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(
        readyTask,
      );

      renderHook(
        () => useSubConversationTaskPolling(taskId, parentConversationId),
        { wrapper },
      );

      await waitFor(() => {
        expect(mockInvalidateQueries).toHaveBeenCalled();
      });

      expect(mockInvalidateQueries).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", parentConversationId],
      });
    });
  });

  describe("when task status is ERROR", () => {
    it("clears subConversationTaskId from localStorage and store on error", async () => {
      const parentConversationId = "parent-conv-123";
      const taskId = "task-123";

      const errorTask = makeTask("ERROR", null);
      vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(
        errorTask,
      );

      renderHook(
        () => useSubConversationTaskPolling(taskId, parentConversationId),
        { wrapper },
      );

      await waitFor(() => {
        expect(setConversationState).toHaveBeenCalledWith(
          parentConversationId,
          { subConversationTaskId: null },
        );
      });

      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(null);
      expect(mockInvalidateQueries).not.toHaveBeenCalled();
    });
  });

  describe("when task is still in progress", () => {
    it("does not clear subConversationTaskId when task status is WORKING", async () => {
      const parentConversationId = "parent-conv-123";
      const taskId = "task-123";

      const workingTask = makeTask("WORKING", null);
      vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(
        workingTask,
      );

      renderHook(
        () => useSubConversationTaskPolling(taskId, parentConversationId),
        { wrapper },
      );

      // Wait a bit to ensure useEffect has run
      await waitFor(() => {
        expect(V1ConversationService.getStartTask).toHaveBeenCalled();
      });

      // Should not clear anything while task is in progress
      expect(setConversationState).not.toHaveBeenCalled();
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
    });

    it("does not clear subConversationTaskId when task status is WAITING_FOR_SANDBOX", async () => {
      const parentConversationId = "parent-conv-123";
      const taskId = "task-123";

      const waitingTask = makeTask("WAITING_FOR_SANDBOX", null);
      vi.mocked(V1ConversationService.getStartTask).mockResolvedValue(
        waitingTask,
      );

      renderHook(
        () => useSubConversationTaskPolling(taskId, parentConversationId),
        { wrapper },
      );

      await waitFor(() => {
        expect(V1ConversationService.getStartTask).toHaveBeenCalled();
      });

      expect(setConversationState).not.toHaveBeenCalled();
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
    });
  });

  describe("when parentConversationId is null", () => {
    it("does not clear subConversationTaskId or invalidate queries", () => {
      const taskId = "task-123";

      renderHook(() => useSubConversationTaskPolling(taskId, null), {
        wrapper,
      });

      // Query is disabled when parentConversationId is null, so getStartTask won't be called
      expect(V1ConversationService.getStartTask).not.toHaveBeenCalled();
      expect(setConversationState).not.toHaveBeenCalled();
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
      expect(mockInvalidateQueries).not.toHaveBeenCalled();
    });
  });
});
