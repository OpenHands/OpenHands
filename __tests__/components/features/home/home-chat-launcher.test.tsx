import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import toast from "react-hot-toast";

import { HomeChatLauncher } from "#/components/features/home/home-chat-launcher";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";

const mockNavigate = vi.fn();
const mockUseActiveBackend = vi.fn();
const sendMessageWithAttachments = vi.fn();
const mockClearAllFiles = vi.fn();
const enqueueHomeTaskPendingMessage = vi.fn();
const mockDisplayErrorToast = vi.fn();
const mockUseLlmConfigured = vi.fn();

let mockImages: File[] = [];
let mockFiles: File[] = [];

vi.mock("#/utils/send-message-with-attachments", () => ({
  sendMessageWithAttachments: (...args: unknown[]) =>
    sendMessageWithAttachments(...args),
}));

vi.mock("#/utils/enqueue-home-task-pending-message", () => ({
  enqueueHomeTaskPendingMessage: (...args: unknown[]) =>
    enqueueHomeTaskPendingMessage(...args),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({
    images: mockImages,
    files: mockFiles,
    imagesMarkedUploadAsFile: [],
    clearAllFiles: mockClearAllFiles,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/utils/custom-toast-handlers")>();
  return {
    ...actual,
    displayErrorToast: (...args: unknown[]) => mockDisplayErrorToast(...args),
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/contexts/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: mockNavigate,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => mockUseActiveBackend(),
}));

vi.mock("#/hooks/use-llm-configured", () => ({
  useLlmConfigured: () => mockUseLlmConfigured(),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

// Stub CustomChatInput as a simple button so the test can submit without
// exercising the rich contenteditable / draft-persistence stack — those are
// covered by their own unit tests. Pressing the stub button is the same
// signal: "user submitted `hello world`".
vi.mock("#/components/features/chat/custom-chat-input", () => ({
  CustomChatInput: ({
    onSubmit,
    disabled,
  }: {
    onSubmit: (msg: string) => void;
    disabled?: boolean;
  }) => (
    <button
      type="button"
      data-testid="stub-chat-submit"
      disabled={disabled}
      onClick={() => onSubmit("hello world")}
    >
      stub submit
    </button>
  ),
}));

// Stub the selection dialogs. We mirror the real component's contract:
// `onConfirm(selection)` is followed by `onClose()` so the parent's pending
// state is set and the dialog disappears.
vi.mock("#/components/features/home/open-workspace-dialog", () => ({
  OpenWorkspaceDialog: ({
    isOpen,
    onClose,
    onConfirm,
  }: {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (w: { id: string; name: string; path: string }) => void;
  }) =>
    isOpen ? (
      <button
        type="button"
        data-testid="stub-workspace-dialog-confirm"
        onClick={() => {
          onConfirm({ id: "/p/app", name: "app", path: "/p/app" });
          onClose();
        }}
      >
        confirm
      </button>
    ) : null,
}));

vi.mock("#/components/features/home/open-repository-dialog", () => ({
  OpenRepositoryDialog: ({
    isOpen,
    onClose,
    onConfirm,
  }: {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (s: {
      repository: {
        id: string;
        full_name: string;
        git_provider: "github";
        is_public: boolean;
      };
      branch: { name: string };
      provider: "github" | null;
    }) => void;
  }) =>
    isOpen ? (
      <button
        type="button"
        data-testid="stub-repo-dialog-confirm"
        onClick={() => {
          onConfirm({
            repository: {
              id: "1",
              full_name: "org/repo",
              git_provider: "github",
              is_public: true,
            },
            branch: { name: "main" },
            provider: "github",
          });
          onClose();
        }}
      >
        confirm
      </button>
    ) : null,
}));

// HomeGitControlBarPreview pulls in settings + provider hooks we don't care
// about for these tests. It's purely presentational once a selection is
// confirmed, so a thin stub is sufficient.
vi.mock("#/components/features/home/home-git-control-bar-preview", () => ({
  HomeGitControlBarPreview: ({
    workspaceMode,
    backendKind,
    onWorkspaceModeChange,
  }: {
    workspaceMode: "local_repo" | "new_worktree";
    backendKind: "local" | "cloud";
    onWorkspaceModeChange: (mode: "local_repo" | "new_worktree") => void;
  }) => (
    <div data-testid="stub-git-control-bar-preview">
      <span data-testid="stub-workspace-mode">
        {backendKind}:{workspaceMode}
      </span>
      <button
        type="button"
        data-testid="stub-workspace-mode-new-worktree"
        onClick={() => onWorkspaceModeChange("new_worktree")}
      >
        New Worktree
      </button>
    </div>
  ),
}));

// Stub the picker modal: pressing it selects one plugin then closes, mirroring
// the real modal's `onChange` + `onClose` contract. The picker catalog itself
// is covered by plugin-picker.test.tsx.
vi.mock("#/components/features/plugins/plugin-picker-modal", () => ({
  PluginPickerModal: ({
    onChange,
    onClose,
  }: {
    onChange: (next: { source: string; ref: null; repo_path: null }[]) => void;
    onClose: () => void;
  }) => (
    <button
      type="button"
      data-testid="stub-plugin-pick"
      onClick={() => {
        onChange([{ source: "github:o/a", ref: null, repo_path: null }]);
        onClose();
      }}
    >
      pick
    </button>
  ),
}));

const renderLauncher = () =>
  render(<HomeChatLauncher />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: {
              queries: { retry: false },
              mutations: { retry: false },
            },
          })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });

function makeConversationResponse(
  overrides: Record<string, unknown> = {},
): never {
  return {
    id: "conv-abc",
    created_by_user_id: null,
    status: "READY",
    detail: null,
    app_conversation_id: "conv-abc",
    agent_server_url: "http://agent-server.local",
    request: { initial_message: undefined, plugins: null },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  } as never;
}

const localBackend = {
  backend: {
    id: "local-id",
    name: "Local",
    host: "http://localhost",
    apiKey: "test",
    kind: "local" as const,
  },
  orgId: null,
};

const cloudBackend = {
  backend: {
    id: "cloud-id",
    name: "Cloud",
    host: "https://cloud",
    apiKey: "test",
    kind: "cloud" as const,
  },
  orgId: null,
};

describe("HomeChatLauncher", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
    mockImages = [];
    mockFiles = [];
    mockUseActiveBackend.mockReturnValue(localBackend);
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: true,
      isLoading: false,
    });
    enqueueHomeTaskPendingMessage.mockResolvedValue(undefined);
    sendMessageWithAttachments.mockResolvedValue({
      text: "hello world",
      content: "hello world",
      imageUrls: ["data:image/png;base64,abc"],
      fileUrls: [],
      timestamp: "2020-01-01T00:00:00.000Z",
    });
    vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    });
  });

  afterEach(() => {
    toast.remove();
  });

  it("creates a conversation with just the typed query and navigates when no workspace is selected", async () => {
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      "hello world",
      undefined,
      undefined,
      null,
      undefined,
      undefined,
      undefined,
      undefined,
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-abc"),
    );
  });

  it("disables the chat input and won't create a conversation when no LLM is configured", async () => {
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: false,
      isLoading: false,
    });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();

    // The send is disabled, so the agent can't be handed a request it can't run.
    expect(screen.getByTestId("stub-chat-submit")).toBeDisabled();
    expect(createSpy).not.toHaveBeenCalled();
  });

  it("passes the picked workspace path as working_dir on a local backend", async () => {
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-ws" }),
      );

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      "hello world",
      undefined,
      undefined,
      null,
      "/p/app",
      "local_repo",
      undefined,
      undefined,
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-ws"),
    );
  });

  it("passes the picked workspace path with new-worktree mode when selected", async () => {
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-wt" }),
      );

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );
    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:local_repo",
    );

    await user.click(screen.getByTestId("stub-workspace-mode-new-worktree"));
    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:new_worktree",
    );
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      "hello world",
      undefined,
      undefined,
      null,
      "/p/app",
      "new_worktree",
      undefined,
      undefined,
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-wt"),
    );
  });

  it("disables the local workspace launcher when the agent server is too old", async () => {
    vi.spyOn(WorkspacesService, "listWorkspaces").mockRejectedValue({
      code: "AGENT_SERVER_VERSION_TOO_OLD",
      feature: "workspaces",
      requiredVersion: "1.23.0",
      actualVersion: "1.22.1",
    });

    renderLauncher();
    const user = userEvent.setup();
    await waitFor(() =>
      expect(screen.getByTestId("open-workspace-button")).toBeDisabled(),
    );
    const button = screen.getByTestId("open-workspace-button");
    await user.hover(button.parentElement ?? button);

    expect(
      await screen.findByText("HOME$WORKSPACES_UNSUPPORTED_AGENT_SERVER"),
    ).toBeInTheDocument();
  });

  it("passes the picked repository + branch payload on a cloud backend", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-repo" }),
      );

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-repository-button"));
    await user.click(await screen.findByTestId("stub-repo-dialog-confirm"));
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      "hello world",
      undefined,
      undefined,
      {
        selected_repository: "org/repo",
        selected_branch: "main",
        git_provider: "github",
      },
      undefined,
      undefined,
      undefined,
      undefined,
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-repo"),
    );
  });

  it("does not pass query to createConversation when attachments are present", async () => {
    mockImages = [new File(["x"], "shot.png", { type: "image/png" })];
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      null,
      undefined,
      undefined,
      undefined,
      undefined,
    );
    await waitFor(() =>
      expect(sendMessageWithAttachments).toHaveBeenCalledTimes(1),
    );
    expect(mockClearAllFiles).toHaveBeenCalled();
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-abc"),
    );
  });

  it("surfaces a toast and skips navigation when conversation creation fails", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockRejectedValue(new Error("Network down"));

    renderLauncher();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(mockDisplayErrorToast).toHaveBeenCalled());
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("enqueues an optimistic pending message when cloud returns a start task", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({
          id: "start-task-1",
          app_conversation_id: null,
        }),
      );

    renderLauncher();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(enqueueHomeTaskPendingMessage).toHaveBeenCalledWith({
        conversationId: "task-start-task-1",
        text: "hello world",
        images: [],
        imagesMarkedUploadAsFile: [],
      }),
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        "/conversations/task-start-task-1",
      ),
    );
  });

  it("defers attachments and enqueues an optimistic pending message for cloud start tasks", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    mockImages = [new File(["x"], "shot.png", { type: "image/png" })];
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({
          id: "start-task-2",
          app_conversation_id: null,
        }),
      );

    renderLauncher();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      undefined,
      undefined,
      undefined,
      null,
      undefined,
      undefined,
      undefined,
      undefined,
    );
    expect(sendMessageWithAttachments).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(enqueueHomeTaskPendingMessage).toHaveBeenCalledWith({
        conversationId: "task-start-task-2",
        text: "hello world",
        images: mockImages,
        imagesMarkedUploadAsFile: [],
      }),
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        "/conversations/task-start-task-2",
      ),
    );
  });

  it("attaches the picked plugins to the created conversation", async () => {
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-plugin-picker"));
    await user.click(await screen.findByTestId("stub-plugin-pick"));
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      "hello world",
      undefined,
      [{ source: "github:o/a", ref: null, repo_path: null }],
      null,
      undefined,
      undefined,
      undefined,
      undefined,
    );
  });
});
