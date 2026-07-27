import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useNewConversationCommand } from "#/hooks/mutation/use-new-conversation-command";
import * as telemetry from "#/services/telemetry";

const mockNavigate = vi.fn();

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/conversations/conv-123",
    conversationId: "conv-123",
    isNavigating: false,
    navigate: mockNavigate,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

const { mockToast } = vi.hoisted(() => {
  const mockToast = Object.assign(vi.fn(), {
    loading: vi.fn(),
    dismiss: vi.fn(),
  });
  return { mockToast };
});

vi.mock("react-hot-toast", () => ({
  default: mockToast,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: { position: "top-right" },
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { email: "user@example.com", user_consents_to_analytics: true },
  }),
}));

const mockConversation = {
  id: "conv-123",
  title: "Test Conversation",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  sandbox_id: "sandbox-abc",
  conversation_version: "V1" as const,
};

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: mockConversation,
  }),
}));

function makeStartTask(overrides: Record<string, unknown> = {}) {
  return {
    id: "task-789",
    created_by_user_id: null,
    status: "READY",
    detail: null,
    app_conversation_id: "new-conv-999",
    agent_server_url: "http://agent-server.local",
    request: {
      initial_message: null,
      processors: [],
      llm_model: null,
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      suggested_task: null,
      title: null,
      trigger: null,
      pr_number: [],
      parent_conversation_id: null,
      agent_type: "default",
    },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

describe("useNewConversationCommand", () => {
  let queryClient: QueryClient;
  let captureMock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    captureMock = vi
      .spyOn(telemetry, "trackEvent")
      .mockResolvedValue(undefined);
    queryClient = new QueryClient({
      defaultOptions: { mutations: { retry: false } },
    });
  });

  afterEach(() => {
    captureMock.mockRestore();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("calls createConversation and navigates on success", async () => {
    const readyTask = makeStartTask();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(readyTask as never);

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalled();
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/new-conv-999");
    });
  });

  it("throws when the start task ends in ERROR", async () => {
    const errorTask = makeStartTask({
      status: "ERROR",
      detail: "Setup failed",
      app_conversation_id: null,
    });

    vi.spyOn(AgentServerConversationService, "createConversation").mockResolvedValue(
      errorTask as never,
    );

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await expect(result.current.mutateAsync()).rejects.toThrow("Setup failed");
  });

  it("navigates to /conversations/task-{id} for a cloud WORKING task without app_conversation_id", async () => {
    const workingTask = makeStartTask({
      status: "WORKING",
      detail: null,
      app_conversation_id: null,
    });

    vi.spyOn(AgentServerConversationService, "createConversation").mockResolvedValue(
      workingTask as never,
    );

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      // Format matches OpenHands' cloud pattern: useTaskPolling unwraps
      // `task-{uuid}` and polls until READY, then redirects.
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/task-task-789");
    });
  });

  it("invalidates conversation list queries on success", async () => {
    const readyTask = makeStartTask();

    vi.spyOn(AgentServerConversationService, "createConversation").mockResolvedValue(
      readyTask as never,
    );

    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["user", "conversations"],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["v1-batch-get-app-conversations"],
      });
    });
  });

  it("forwards the active conversation's sandbox_id so /new reuses the same runtime", async () => {
    // Arrange
    const readyTask = makeStartTask();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(readyTask as never);

    // Act
    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });
    await result.current.mutateAsync();

    // Assert — sandbox_id is the 9th positional argument; parent_conversation_id
    // and agent_type stay undefined because /new is NOT a sub-conversation.
    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        "sandbox-abc",
      );
    });
  });

  it("shows a loading toast and dismisses it on success", async () => {
    const readyTask = makeStartTask();

    vi.spyOn(AgentServerConversationService, "createConversation").mockResolvedValue(
      readyTask as never,
    );

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(mockToast.loading).toHaveBeenCalledWith(
        "CONVERSATION$CLEARING",
        expect.objectContaining({ id: "clear-conversation" }),
      );
      expect(mockToast.dismiss).toHaveBeenCalledWith("clear-conversation");
    });
  });

  it("emits conversation_created on success with the /new no-context payload", async () => {
    const readyTask = makeStartTask();
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockResolvedValue(readyTask as never);

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(captureMock).toHaveBeenCalledWith(
        "conversation_created",
        expect.objectContaining({
          conversation_id: "new-conv-999",
          task_id: "task-789",
          is_start_task: false,
          has_repository: false,
          has_workspace: false,
          has_initial_query: false,
          has_parent_conversation: false,
          entry_point: "new_command",
        }),
      );
    });
  });
});
