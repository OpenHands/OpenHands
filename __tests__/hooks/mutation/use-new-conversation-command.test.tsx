import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useNewConversationCommand } from "#/hooks/mutation/use-new-conversation-command";
import {
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";

const {
  mockCreateConversation,
  mockNavigate,
  mockToast,
  mockDisplaySuccessToast,
  mockDisplayErrorToast,
} = vi.hoisted(() => {
  const toast = Object.assign(vi.fn(), {
    loading: vi.fn(),
    dismiss: vi.fn(),
  });
  return {
    mockCreateConversation: vi.fn(),
    mockNavigate: vi.fn(),
    mockToast: toast,
    mockDisplaySuccessToast: vi.fn(),
    mockDisplayErrorToast: vi.fn(),
  };
});

interface MockActiveBackend {
  backend: { id: string; kind: "local" | "cloud" };
  orgId: string | null;
}

const mockUseActiveBackend = vi.fn<() => MockActiveBackend>(() => ({
  backend: { id: "cloud-1", kind: "cloud" as const },
  orgId: null,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));

vi.mock("#/api/backend-registry/active-store", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("#/api/backend-registry/active-store")
    >();
  return {
    ...actual,
    getActiveBackend: () => mockUseActiveBackend(),
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
  displaySuccessToast: mockDisplaySuccessToast,
  displayErrorToast: mockDisplayErrorToast,
  TOAST_OPTIONS: { position: "top-right" },
}));

interface MockConversation {
  id: string;
  title: string;
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: "github" | null;
  sandbox_id: string | null;
  workspace: { working_dir: string };
  conversation_version: "V1";
  launched_agent_profile: { agent_profile_id: string } | null;
}

const mockConversation: MockConversation = {
  id: "conv-123",
  title: "Test Conversation",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  sandbox_id: "sandbox-abc",
  workspace: { working_dir: "C:/workspace/source" },
  conversation_version: "V1",
  launched_agent_profile: null,
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
    mockToast.loading.mockReturnValue("new-toast-1");
    mockCreateConversation.mockResolvedValue(readyConversation);
    queryClient = new QueryClient({
      defaultOptions: { mutations: { retry: false } },
    });
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    });
    mockConversation.selected_repository = null;
    mockConversation.selected_branch = null;
    mockConversation.git_provider = null;
    mockConversation.sandbox_id = "sandbox-abc";
    mockConversation.launched_agent_profile = null;
    removeStoredConversationMetadata("conv-123");
  });

  it("reuses the exact local workspace without creating a parent", async () => {
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "local-1", kind: "local" },
      orgId: null,
    });
    setStoredConversationMetadata("conv-123", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: "C:/workspace/source",
      workspace_mode: "local_repo",
    });

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    expect(mockCreateConversation).toHaveBeenCalledWith({
      workingDir: "C:/workspace/source",
      workspaceMode: "local_repo",
      entryPoint: "new_command",
    });
    expect(mockCreateConversation.mock.calls[0][0]).not.toHaveProperty(
      "parentConversationId",
    );
  });

  it("preserves local launch metadata without allocating another worktree", async () => {
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "local-1", kind: "local" },
      orgId: null,
    });
    mockConversation.selected_repository = "org/repo";
    mockConversation.selected_branch = "feature";
    mockConversation.git_provider = "github";
    mockConversation.launched_agent_profile = {
      agent_profile_id: "profile-1",
    };
    setStoredConversationMetadata("conv-123", {
      selected_repository: "org/repo",
      selected_branch: "feature",
      git_provider: "github",
      selected_workspace: "C:/workspace/source",
      workspace_mode: "new_worktree",
      plugins: [
        { source: "github:org/plugin", ref: "main", repo_path: null },
      ],
    });

    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    await result.current.mutateAsync();

    expect(mockCreateConversation).toHaveBeenCalledWith({
      repository: {
        name: "org/repo",
        branch: "feature",
        gitProvider: "github",
      },
      plugins: [
        { source: "github:org/plugin", ref: "main", repo_path: null },
      ],
      agentProfileId: "profile-1",
      workingDir: "C:/workspace/source",
      workspaceMode: "local_repo",
      entryPoint: "new_command",
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
    await waitFor(() => {
      expect(mockToast.dismiss).toHaveBeenCalledWith("new-toast-1");
      expect(mockDisplayErrorToast).toHaveBeenCalledOnce();
    });
    expect(mockDisplaySuccessToast).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("does not navigate or report late success after a backend switch", async () => {
    let resolveCreation!: (value: typeof readyConversation) => void;
    mockCreateConversation.mockReturnValue(
      new Promise((resolve) => {
        resolveCreation = resolve;
      }),
    );
    const { result } = renderHook(() => useNewConversationCommand(), {
      wrapper,
    });

    const pending = result.current.mutateAsync();
    mockUseActiveBackend.mockReturnValue({
      backend: { id: "cloud-2", kind: "cloud" },
      orgId: null,
    });
    resolveCreation(readyConversation);
    await pending;

    expect(mockToast.dismiss).toHaveBeenCalledWith("new-toast-1");
    expect(mockDisplaySuccessToast).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
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
        expect.objectContaining({ position: "top-right" }),
      );
      expect(mockToast.dismiss).toHaveBeenCalledWith("new-toast-1");
    });
  });
});
