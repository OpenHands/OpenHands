import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import type { Conversation } from "#/api/open-hands.types";

// Mock dependencies
vi.mock("#/stores/conversation-store");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/mutation/use-create-conversation");
vi.mock("#/utils/conversation-local-storage");
vi.mock("#/utils/custom-toast-handlers");
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

const mockSetConversationMode = vi.fn();
const mockSetSubConversationTaskId = vi.fn();
const mockCreateConversation = vi.fn();

// Helper function to create properly typed mock return values
function asMockReturnValue<T>(value: Partial<T>): T {
  return value as T;
}

function makeConversation(overrides?: Partial<Conversation>): Conversation {
  return {
    conversation_id: "conv-123",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: null,
    url: null,
    session_api_key: null,
    conversation_version: "V1",
    sub_conversation_ids: [],
    ...overrides,
  } as Conversation;
}

describe("useHandlePlanClick", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useConversationStore).mockReturnValue({
      setConversationMode: mockSetConversationMode,
      setSubConversationTaskId: mockSetSubConversationTaskId,
      subConversationTaskId: null,
    });

    vi.mocked(useActiveConversation).mockReturnValue(
      asMockReturnValue<ReturnType<typeof useActiveConversation>>({
        data: makeConversation(),
        isLoading: false,
        isPending: false,
        isError: false,
        error: null,
        refetch: vi.fn(),
      }),
    );

    vi.mocked(useCreateConversation).mockReturnValue(
      asMockReturnValue<ReturnType<typeof useCreateConversation>>({
        mutate: mockCreateConversation,
        isPending: false,
        isSuccess: false,
        isError: false,
        error: null,
      }),
    );

    vi.mocked(getConversationState).mockReturnValue({
      selectedTab: "editor",
      rightPanelShown: true,
      unpinnedTabs: [],
      subConversationTaskId: null,
      conversationMode: "code",
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("localStorage restoration", () => {
    it("restores subConversationTaskId from localStorage when conversation loads", () => {
      const conversationId = "conv-123";
      const storedTaskId = "task-456";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ conversation_id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      vi.mocked(getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        subConversationTaskId: storedTaskId,
        conversationMode: "code",
      });

      renderHook(() => useHandlePlanClick());

      expect(getConversationState).toHaveBeenCalledWith(conversationId);
      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(storedTaskId);
    });

    it("does not restore subConversationTaskId if it already exists in store", () => {
      const conversationId = "conv-123";
      const storedTaskId = "task-456";
      const existingTaskId = "task-789";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ conversation_id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      vi.mocked(useConversationStore).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useConversationStore>>({
          setConversationMode: mockSetConversationMode,
          setSubConversationTaskId: mockSetSubConversationTaskId,
          subConversationTaskId: existingTaskId,
        }),
      );

      vi.mocked(getConversationState).mockReturnValue({
        selectedTab: "editor",
        rightPanelShown: true,
        unpinnedTabs: [],
        subConversationTaskId: storedTaskId,
        conversationMode: "code",
      });

      renderHook(() => useHandlePlanClick());

      expect(getConversationState).toHaveBeenCalledWith(conversationId);
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
    });

    it("does not restore subConversationTaskId when conversation is not loaded", () => {
      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: undefined,
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      renderHook(() => useHandlePlanClick());

      expect(getConversationState).not.toHaveBeenCalled();
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
    });
  });

  describe("plan creation prevention", () => {
    it("prevents plan creation when subConversationTaskId exists in store", () => {
      const taskId = "task-123";

      vi.mocked(useConversationStore).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useConversationStore>>({
          setConversationMode: mockSetConversationMode,
          setSubConversationTaskId: mockSetSubConversationTaskId,
          subConversationTaskId: taskId,
        }),
      );

      const { result } = renderHook(() => useHandlePlanClick());

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).not.toHaveBeenCalled();
    });

    it("prevents plan creation when conversation has existing sub_conversation_ids", () => {
      vi.mocked(useActiveConversation).mockReturnValue({
        data: makeConversation({
          sub_conversation_ids: ["sub-conv-1"],
        }),
        isLoading: false,
        isPending: false,
        isError: false,
        error: null,
        refetch: vi.fn(),
      } as Partial<ReturnType<typeof useActiveConversation>> as ReturnType<
        typeof useActiveConversation
      >);

      const { result } = renderHook(() => useHandlePlanClick());

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).not.toHaveBeenCalled();
    });

    it("prevents plan creation when conversation_id is missing", () => {
      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: undefined,
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      const { result } = renderHook(() => useHandlePlanClick());

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).not.toHaveBeenCalled();
    });
  });

  describe("plan creation and persistence", () => {
    it("creates plan conversation and persists subConversationTaskId to localStorage", () => {
      const conversationId = "conv-123";
      const taskId = "task-789";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ conversation_id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      const { result } = renderHook(() => useHandlePlanClick());

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).toHaveBeenCalledWith(
        {
          parentConversationId: conversationId,
          agentType: "plan",
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
        }),
      );

      // Simulate successful conversation creation
      const onSuccessCallback = mockCreateConversation.mock.calls[0][1]
        .onSuccess as (data: { v1_task_id?: string }) => void;

      act(() => {
        onSuccessCallback({ v1_task_id: taskId });
      });

      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(taskId);
      expect(setConversationState).toHaveBeenCalledWith(conversationId, {
        subConversationTaskId: taskId,
      });
      expect(displaySuccessToast).toHaveBeenCalled();
    });

    it("does not persist subConversationTaskId when v1_task_id is missing", () => {
      const conversationId = "conv-123";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ conversation_id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      const { result } = renderHook(() => useHandlePlanClick());

      act(() => {
        result.current.handlePlanClick();
      });

      const onSuccessCallback = mockCreateConversation.mock.calls[0][1]
        .onSuccess as (data: { v1_task_id?: string }) => void;

      act(() => {
        onSuccessCallback({});
      });

      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
      expect(setConversationState).not.toHaveBeenCalled();
    });
  });
});
