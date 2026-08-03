import React from "react";
import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useHandlePlanClick } from "#/hooks/use-handle-plan-click";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { useActiveBackend } from "#/contexts/active-backend-context";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { LOCAL_PLANNER_PARENT_TAG_KEY } from "#/utils/plan-file";

// Mock dependencies
vi.mock("#/stores/conversation-store");
vi.mock("#/hooks/query/use-active-conversation");
vi.mock("#/hooks/mutation/use-create-conversation");
vi.mock("#/hooks/query/use-sub-conversations");
vi.mock("#/utils/conversation-local-storage");
vi.mock("#/utils/custom-toast-handlers");
vi.mock("#/contexts/active-backend-context");
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      createLocalPlanningConversation: vi.fn(),
    },
  }),
);
vi.mock("#/api/conversation-metadata-store", () => ({
  getStoredConversationMetadata: vi.fn(),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
  initReactI18next: {
    type: "3rdParty",
    init: () => {},
  },
}));

const mockSetConversationMode = vi.fn();
const mockSetSubConversationTaskId = vi.fn();
const mockSetLocalPlanningConversationId = vi.fn();
const mockCreateConversation = vi.fn();

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

function renderPlanHook() {
  return renderHook(() => useHandlePlanClick(), { wrapper: createWrapper() });
}

// Helper function to create properly typed mock return values
function asMockReturnValue<T>(value: Partial<T>): T {
  return value as T;
}

function makeConversation(
  overrides?: Partial<AppConversation>,
): AppConversation {
  return {
    id: "conv-123",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: null,
    conversation_url: null,
    session_api_key: null,
    conversation_version: "V1",
    sub_conversation_ids: [],
    ...overrides,
  } as AppConversation;
}

/** Tags a fetched sub-conversation as the planner helper for `parentId`. */
function makeTaggedPlannerConversation(
  id: string,
  parentId: string,
): AppConversation {
  return makeConversation({
    id,
    tags: { [LOCAL_PLANNER_PARENT_TAG_KEY]: parentId },
  });
}

describe("useHandlePlanClick", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useActiveBackend).mockReturnValue({
      backend: { kind: "cloud" },
    } as ReturnType<typeof useActiveBackend>);
    vi.mocked(getStoredConversationMetadata).mockReturnValue(null);

    vi.mocked(useConversationStore).mockReturnValue({
      setConversationMode: mockSetConversationMode,
      setSubConversationTaskId: mockSetSubConversationTaskId,
      subConversationTaskId: null,
      setLocalPlanningConversationId: mockSetLocalPlanningConversationId,
      localPlanningConversationId: null,
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

    vi.mocked(useSubConversations).mockReturnValue(
      asMockReturnValue<ReturnType<typeof useSubConversations>>({
        data: [],
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
      selectedTab: "files",
      unpinnedTabs: [],
      subConversationTaskId: null,
      conversationMode: "code",
      draftMessage: null,
      filesTabDiffView: null,
      filesTabContentViewMode: "rich",
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
          data: makeConversation({ id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      vi.mocked(getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        subConversationTaskId: storedTaskId,
        conversationMode: "code",
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });

      renderPlanHook();

      expect(getConversationState).toHaveBeenCalledWith(conversationId);
      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(storedTaskId);
    });

    it("does not restore subConversationTaskId if it already exists in store", () => {
      const conversationId = "conv-123";
      const storedTaskId = "task-456";
      const existingTaskId = "task-789";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ id: conversationId }),
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
          setLocalPlanningConversationId: mockSetLocalPlanningConversationId,
          localPlanningConversationId: null,
        }),
      );

      vi.mocked(getConversationState).mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        subConversationTaskId: storedTaskId,
        conversationMode: "code",
        draftMessage: null,
        filesTabDiffView: null,
        filesTabContentViewMode: "rich",
      });

      renderPlanHook();

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

      renderPlanHook();

      expect(getConversationState).not.toHaveBeenCalled();
      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
    });
  });

  describe("local planner conversations", () => {
    it("restores local planning conversation id from metadata", () => {
      vi.mocked(getStoredConversationMetadata).mockReturnValue({
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        local_planning_conversation_id: "plan-conv-1",
      });

      renderPlanHook();

      expect(mockSetLocalPlanningConversationId).toHaveBeenCalledWith(
        "plan-conv-1",
      );
    });

    it("creates a local planning conversation on local backends", async () => {
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      vi.mocked(
        AgentServerConversationService.createLocalPlanningConversation,
      ).mockResolvedValue(makeConversation({ id: "plan-conv-1" }));

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      await waitFor(() => {
        expect(
          AgentServerConversationService.createLocalPlanningConversation,
        ).toHaveBeenCalledWith("conv-123");
      });
      await waitFor(() => {
        expect(mockSetLocalPlanningConversationId).toHaveBeenCalledWith(
          "plan-conv-1",
        );
      });
      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).not.toHaveBeenCalled();
      expect(displaySuccessToast).toHaveBeenCalled();
    });

    it("recovers the planner from the server when browser storage was cleared", () => {
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      // Storage loss: no metadata hint, only the server-derived relationship.
      vi.mocked(getStoredConversationMetadata).mockReturnValue(null);
      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ sub_conversation_ids: ["plan-conv-1"] }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );
      vi.mocked(useSubConversations).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useSubConversations>>({
          data: [makeTaggedPlannerConversation("plan-conv-1", "conv-123")],
        }),
      );

      renderPlanHook();

      expect(mockSetLocalPlanningConversationId).toHaveBeenCalledWith(
        "plan-conv-1",
      );
    });

    it("adopts the server-reported planner instead of creating a second hidden one", () => {
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      vi.mocked(getStoredConversationMetadata).mockReturnValue(null);
      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ sub_conversation_ids: ["plan-conv-1"] }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );
      vi.mocked(useSubConversations).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useSubConversations>>({
          data: [makeTaggedPlannerConversation("plan-conv-1", "conv-123")],
        }),
      );

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      expect(
        AgentServerConversationService.createLocalPlanningConversation,
      ).not.toHaveBeenCalled();
    });

    it("does not adopt an unrelated non-planner child conversation as the planner", async () => {
      // Regression: sub_conversation_ids is the generic child list — an
      // existing, non-planner child (e.g. a delegated sub-agent) must not be
      // mistaken for the planner just because it's present in the list.
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      vi.mocked(getStoredConversationMetadata).mockReturnValue(null);
      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ sub_conversation_ids: ["other-conv-1"] }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );
      // Fetched, but untagged for this parent — not the planner.
      vi.mocked(useSubConversations).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useSubConversations>>({
          data: [makeConversation({ id: "other-conv-1", tags: null })],
        }),
      );
      vi.mocked(
        AgentServerConversationService.createLocalPlanningConversation,
      ).mockResolvedValue(makeConversation({ id: "plan-conv-1" }));

      const { result } = renderPlanHook();

      // The unrelated child must never be restored as the planner.
      expect(mockSetLocalPlanningConversationId).not.toHaveBeenCalledWith(
        "other-conv-1",
      );

      act(() => {
        result.current.handlePlanClick();
      });

      // With no tagged planner found, clicking Plan creates a real one
      // instead of silently adopting the unrelated child.
      await waitFor(() => {
        expect(
          AgentServerConversationService.createLocalPlanningConversation,
        ).toHaveBeenCalledWith("conv-123");
      });
    });

    it("resets to code mode and shows an error toast when local planner creation fails", async () => {
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      vi.mocked(
        AgentServerConversationService.createLocalPlanningConversation,
      ).mockRejectedValue(new Error("boom"));

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");

      await waitFor(() => {
        expect(mockSetConversationMode).toHaveBeenCalledWith("code");
      });
      expect(displayErrorToast).toHaveBeenCalled();
      expect(mockSetLocalPlanningConversationId).not.toHaveBeenCalled();
    });

    it("does not create a duplicate local planning conversation", () => {
      vi.mocked(useActiveBackend).mockReturnValue({
        backend: { kind: "local" },
      } as ReturnType<typeof useActiveBackend>);
      vi.mocked(useConversationStore).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useConversationStore>>({
          setConversationMode: mockSetConversationMode,
          setSubConversationTaskId: mockSetSubConversationTaskId,
          subConversationTaskId: null,
          setLocalPlanningConversationId: mockSetLocalPlanningConversationId,
          localPlanningConversationId: "plan-conv-1",
        }),
      );

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      expect(
        AgentServerConversationService.createLocalPlanningConversation,
      ).not.toHaveBeenCalled();
      expect(mockCreateConversation).not.toHaveBeenCalled();
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
          setLocalPlanningConversationId: mockSetLocalPlanningConversationId,
          localPlanningConversationId: null,
        }),
      );

      const { result } = renderPlanHook();

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

      const { result } = renderPlanHook();

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

      const { result } = renderPlanHook();

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
          data: makeConversation({ id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      expect(mockSetConversationMode).toHaveBeenCalledWith("plan");
      expect(mockCreateConversation).toHaveBeenCalledWith(
        {
          parentConversationId: conversationId,
          agentType: "plan",
          entryPoint: "plan_sub_conversation",
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
        }),
      );

      // Simulate successful conversation creation
      const onSuccessCallback = mockCreateConversation.mock.calls[0][1]
        .onSuccess as (data: { task_id?: string }) => void;

      act(() => {
        onSuccessCallback({ task_id: taskId });
      });

      expect(mockSetSubConversationTaskId).toHaveBeenCalledWith(taskId);
      expect(setConversationState).toHaveBeenCalledWith(conversationId, {
        subConversationTaskId: taskId,
      });
      expect(displaySuccessToast).toHaveBeenCalled();
    });

    it("does not persist subConversationTaskId when task_id is missing", () => {
      const conversationId = "conv-123";

      vi.mocked(useActiveConversation).mockReturnValue(
        asMockReturnValue<ReturnType<typeof useActiveConversation>>({
          data: makeConversation({ id: conversationId }),
          isLoading: false,
          isPending: false,
          isError: false,
          error: null,
          refetch: vi.fn(),
        }),
      );

      const { result } = renderPlanHook();

      act(() => {
        result.current.handlePlanClick();
      });

      const onSuccessCallback = mockCreateConversation.mock.calls[0][1]
        .onSuccess as (data: { task_id?: string }) => void;

      act(() => {
        onSuccessCallback({});
      });

      expect(mockSetSubConversationTaskId).not.toHaveBeenCalled();
      expect(setConversationState).not.toHaveBeenCalled();
    });
  });
});
