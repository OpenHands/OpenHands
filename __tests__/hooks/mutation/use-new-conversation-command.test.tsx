import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useNewConversationCommand } from "#/hooks/mutation/use-new-conversation-command";

const { mockCreateConversation, mockNavigate, mockToast } = vi.hoisted(() => {
  const toast = Object.assign(vi.fn(), {
    loading: vi.fn(),
    dismiss: vi.fn(),
  });
  return {
    mockCreateConversation: vi.fn(),
    mockNavigate: vi.fn(),
    mockToast: toast,
  };
});

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({ mutateAsync: mockCreateConversation }),
}));

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

vi.mock("react-hot-toast", () => ({
  default: mockToast,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: { position: "top-right" },
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

const readyConversation = {
  conversation_id: "new-conv-999",
  session_api_key: null,
  url: "http://agent-server.local",
  task_id: "task-789",
};

describe("useNewConversationCommand", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCreateConversation.mockResolvedValue(readyConversation);
    queryClient = new QueryClient({
      defaultOptions: { mutations: { retry: false } },
    });
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("uses the shared New Chat creation path and navigates on success", async () => {
    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(mockCreateConversation).toHaveBeenCalledWith({
        sandboxId: "sandbox-abc",
        entryPoint: "new_command",
      });
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/new-conv-999");
    });
  });

  it("surfaces errors from the shared creation path", async () => {
    mockCreateConversation.mockRejectedValue(new Error("Setup failed"));

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await expect(result.current.mutateAsync()).rejects.toThrow("Setup failed");
  });

  it("navigates to the shared task route for a cloud conversation still provisioning", async () => {
    mockCreateConversation.mockResolvedValue({
      ...readyConversation,
      conversation_id: "task-task-789",
    });

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/task-task-789");
    });
  });

  it("shows a loading toast and dismisses it on success", async () => {
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
});
