import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { compile } from "@tailwindcss/node";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import toast from "react-hot-toast";

import { HomeChatLauncher } from "#/components/features/home/home-chat-launcher";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AutomationService from "#/api/automation-service/automation-service.api";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import GitService from "#/api/git-service/git-service.api";
import { useHomeStore } from "#/stores/home-store";
import { useModelStore } from "#/stores/model-store";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import { COLOR_THEMES } from "#/themes/color-themes";
import {
  LAST_LOCAL_WORKSPACE_MODE_STORAGE_KEY,
  writeStoredLocalWorkspaceMode,
} from "#/utils/workspace-mode";
import { HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY } from "#/components/features/home/home-recommended-automations.constants";
import { NON_REPO_SUGGESTIONS } from "#/utils/suggestions/non-repo-suggestions";

const mockNavigate = vi.fn();
const mockUseActiveBackend = vi.fn();
const sendMessageWithAttachments = vi.fn();
const mockClearAllFiles = vi.fn();
const mockRestoreMessageToInputIfEmpty = vi.fn();
const mockSetMessageToSend = vi.fn();
const enqueueHomeTaskPendingMessage = vi.fn();
const mockDisplayErrorToast = vi.fn();
const mockUseLlmConfigured = vi.fn();
const mockUseConversationWorkspace = vi.fn();
const mockMarkAutomationSetupHandoff = vi.fn();

let mockImages: File[] = [];
let mockFiles: File[] = [];
let mockIsolated = false;

vi.mock("#/utils/send-message-with-attachments", () => ({
  sendMessageWithAttachments: (...args: unknown[]) =>
    sendMessageWithAttachments(...args),
}));

vi.mock("#/utils/enqueue-home-task-pending-message", () => ({
  enqueueHomeTaskPendingMessage: (...args: unknown[]) =>
    enqueueHomeTaskPendingMessage(...args),
}));

vi.mock("#/api/automation-setup-handoff-store", () => ({
  markAutomationSetupHandoff: (...args: unknown[]) =>
    mockMarkAutomationSetupHandoff(...args),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({
    images: mockImages,
    files: mockFiles,
    imagesMarkedUploadAsFile: [],
    clearAllFiles: mockClearAllFiles,
    restoreMessageToInputIfEmpty: mockRestoreMessageToInputIfEmpty,
    setMessageToSend: mockSetMessageToSend,
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

vi.mock("#/context/navigation-context", () => ({
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

vi.mock("#/hooks/query/use-conversation-workspace", () => ({
  useConversationWorkspace: () => mockUseConversationWorkspace(),
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
    placeholder,
  }: {
    onSubmit: (msg: string) => void;
    disabled?: boolean;
    placeholder?: string;
  }) => (
    <button
      type="button"
      data-testid="stub-chat-submit"
      data-placeholder={placeholder}
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
    workspaceModes,
    backendKind,
    repository,
    onClear,
    onWorkspaceModeChange,
  }: {
    workspaceMode: WorkspaceMode;
    workspaceModes: WorkspaceMode[];
    backendKind: "local" | "cloud";
    repository?: { full_name: string } | null;
    onClear: () => void;
    onWorkspaceModeChange: (mode: WorkspaceMode) => void;
  }) => (
    <div data-testid="stub-git-control-bar-preview">
      <span data-testid="stub-preview-repository">
        {repository?.full_name ?? "none"}
      </span>
      <button
        type="button"
        data-testid="stub-clear-selection"
        onClick={onClear}
      >
        Clear
      </button>
      <span data-testid="stub-workspace-mode">
        {backendKind}:{workspaceMode}
      </span>
      <span data-testid="stub-workspace-modes">{workspaceModes.join(",")}</span>
      <button
        type="button"
        data-testid="stub-workspace-mode-docker-container"
        onClick={() => onWorkspaceModeChange("docker_container")}
      >
        Docker
      </button>
      <button
        type="button"
        data-testid="stub-workspace-mode-new-worktree"
        onClick={() => onWorkspaceModeChange("new_worktree")}
      >
        Worktree
      </button>
      <button
        type="button"
        data-testid="stub-workspace-mode-local-repo"
        onClick={() => onWorkspaceModeChange("local_repo")}
      >
        Local
      </button>
    </div>
  ),
}));

// Stub the picker modal: pressing it selects one plugin then closes, mirroring
// the real modal's `onChange` + `onClose` contract. The picker catalog itself
// is covered by plugin-picker.test.tsx.
vi.mock(
  "#/components/features/automations/recommended-automations-launcher",
  () => ({
    RecommendedAutomationsLauncher: ({
      variant,
      className,
      dismissible,
      onDismiss,
    }: {
      variant?: string;
      className?: string;
      dismissible?: boolean;
      onDismiss?: () => void;
    }) =>
      variant === "rail" ? (
        <div data-testid="recommended-automations-rail" className={className}>
          {dismissible && onDismiss && (
            <button
              type="button"
              data-testid="recommended-automations-rail-dismiss"
              onClick={onDismiss}
            >
              Hide
            </button>
          )}
        </div>
      ) : null,
  }),
);

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

function makeBranch(name: string) {
  return { name, commit_sha: "abc", protected: false };
}

function mockConnectedGitHub() {
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...MOCK_DEFAULT_USER_SETTINGS,
    provider_tokens_set: { github: null },
  });
  vi.spyOn(GitService, "searchRepositoryBranches").mockResolvedValue({
    items: [makeBranch("main")],
    next_page_id: null,
  });
}

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
    mockIsolated = false;
    mockUseActiveBackend.mockReturnValue(localBackend);
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: true,
      isLoading: false,
    });
    mockUseConversationWorkspace.mockImplementation(() => ({
      isolated: mockIsolated,
      unsupportedMessage: mockIsolated ? "isolated-unsupported" : null,
    }));
    enqueueHomeTaskPendingMessage.mockResolvedValue(undefined);
    sendMessageWithAttachments.mockResolvedValue({
      text: "hello world",
      content: "hello world",
      imageUrls: ["data:image/png;base64,abc"],
      fileUrls: [],
      timestamp: "2020-01-01T00:00:00.000Z",
    });
    window.localStorage.removeItem(LAST_LOCAL_WORKSPACE_MODE_STORAGE_KEY);
    vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    });
    // The launcher mounts the pinned/running automation dashboards, whose
    // queries would otherwise fire real axios XHRs into MSW. If such a
    // request is still in flight when the file's jsdom environment is torn
    // down, MSW's XHR interceptor throws `ReferenceError:
    // XMLHttpRequestUpload is not defined` as an unhandled rejection.
    // Mocking the underlying service keeps all automation traffic in-process.
    vi.spyOn(AutomationService, "checkHealth").mockResolvedValue({
      status: "ok",
    });
    vi.spyOn(AutomationService, "getAutomations").mockResolvedValue({
      automations: [],
      total: 0,
    });
  });

  afterEach(() => {
    toast.remove();
    window.localStorage.removeItem(LAST_LOCAL_WORKSPACE_MODE_STORAGE_KEY);
    useModelStore.setState({ homeLlmProfile: null });
  });

  it("compiles the rendered launcher colors for every theme, including the mocked isolated-workspace notice", async () => {
    // Reuse the isolated-backend fixture; this is a consumer/compiler contract,
    // not evidence of a running isolated runtime. Test mode uses the static pill.
    mockIsolated = true;
    const { container } = renderLauncher();
    const activeOption = screen.getByRole("radio", { checked: true });
    const roles = [
      [screen.getByRole("heading", { level: 1 }), "color", "--oh-contrast"],
      [
        screen.getByTestId("home-header-subtitle"),
        "color",
        "--oh-text-tertiary",
      ],
      [activeOption, "color", "--oh-contrast"],
      [screen.getByRole("radio", { checked: false }), "color", "--oh-muted"],
      [
        screen.getByRole("radiogroup"),
        "background-color",
        "--oh-surface-raised",
      ],
      [activeOption, "background-color", "--oh-interactive-hover"],
      [screen.getByRole("status"), "color", "--oh-text-secondary"],
    ] as const;
    expect(screen.getByRole("status")).toHaveTextContent(
      "HOME$ISOLATED_WORKSPACE_NEW",
    );

    const compiler = await compile(
      readFileSync(resolvePath("src/tailwind.css"), "utf8"),
      { base: resolvePath("src"), onDependency: () => {} },
    );
    const css = compiler.build(
      [...container.querySelectorAll("[class]")].flatMap((element) => [
        ...element.classList,
      ]),
    );
    const declaration = (element: Element, property: string) => {
      for (const candidate of element.classList) {
        const selector = `.${candidate.replace(/[^\w-]/g, "\\$&")} {`;
        const start = css.indexOf(selector);
        if (start < 0) continue;
        const rule = css.slice(
          start + selector.length,
          css.indexOf("}", start),
        );
        const value = rule.match(
          new RegExp(`(?:^|[;{])\\s*${property}:\\s*([^;]+);`),
        )?.[1];
        if (value) return value;
      }
      return undefined;
    };
    const colors = roles.map(([element, property, token]) => ({
      value: declaration(element, property),
      token,
      property,
    }));

    for (const [key, theme] of Object.entries(COLOR_THEMES)) {
      const variables: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      const resolveColor = (value: string): string => {
        const reference = /^var\((--[\w-]+)\)$/.exec(value);
        if (!reference) return value;
        const resolved = variables[reference[1]];
        expect(resolved, `${key} ${reference[1]}`).toBeDefined();
        return resolveColor(resolved);
      };
      for (const { value, token, property } of colors) {
        expect(resolveColor(value ?? ""), `${key} ${token} ${property}`).toBe(
          resolveColor(`var(${token})`),
        );
      }
    }
  });

  it("defaults to Code mode and switches to Automate mode", async () => {
    renderLauncher();
    const user = userEvent.setup();

    expect(screen.getByTestId("home-launcher-mode-code")).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.getByTestId("home-composer-actions")).toHaveAttribute(
      "aria-disabled",
      "false",
    );
    expect(screen.getByTestId("open-workspace-button")).toBeEnabled();
    expect(screen.getByTestId("open-plugin-picker")).toBeEnabled();
    expect(screen.getByTestId("stub-chat-submit")).toHaveAttribute(
      "data-placeholder",
      "SUGGESTIONS$WHAT_TO_BUILD",
    );
    await user.click(screen.getByTestId("home-launcher-mode-automate"));

    expect(screen.getByTestId("home-launcher-mode-automate")).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.getByTestId("home-composer-actions")).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getByTestId("open-workspace-button")).toBeDisabled();
    expect(screen.getByTestId("open-plugin-picker")).toBeDisabled();
    expect(screen.getByTestId("stub-chat-submit")).toHaveAttribute(
      "data-placeholder",
      "HOME$AUTOMATE_PROMPT_PLACEHOLDER",
    );
    expect(
      screen.getByTestId("recommended-automations-rail"),
    ).toBeInTheDocument();
  });

  it("creates a conversation with just the typed query and navigates when no workspace is selected", async () => {
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(mockMarkAutomationSetupHandoff).not.toHaveBeenCalled();

    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        initialUserMsg: "hello world",
        metadata: null,
      }),
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
      expect.objectContaining({
        initialUserMsg: "hello world",
        metadata: null,
        workingDirOverride: "/p/app",
        workspaceMode: "local_repo",
      }),
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-ws"),
    );
  });

  it("omits a stale host workspace override on an isolated backend", async () => {
    // A host folder selected while the backend looked like a normal local
    // backend must not be forwarded once the backend advertises isolation: the
    // server rejects it and the user sees an error toast for a selection the
    // launcher already deems unsupported. `isolated` is read per render, so
    // flipping the mocked value and forcing a re-render models the backend
    // changing under the user.
    mockUseConversationWorkspace.mockImplementation(() => ({
      isolated: mockIsolated,
      unsupportedMessage: mockIsolated ? "isolated-unsupported" : null,
    }));
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-iso" }),
      );

    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );

    mockIsolated = true;
    await user.click(screen.getByTestId("stub-workspace-mode-new-worktree"));
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy.mock.calls[0]?.[0]).toMatchObject({
      initialUserMsg: "hello world",
      metadata: null,
      workingDirOverride: undefined,
      workspaceMode: undefined,
    });
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
      expect.objectContaining({
        initialUserMsg: "hello world",
        metadata: null,
        workingDirOverride: "/p/app",
        workspaceMode: "new_worktree",
      }),
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-wt"),
    );
  });

  it("starts a Docker conversation when the server offers Docker", async () => {
    mockUseConversationWorkspace.mockReturnValue({
      isolated: false,
      dockerSelectable: true,
      unsupportedMessage: null,
    });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-docker" }),
      );
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );
    await user.click(
      screen.getByTestId("stub-workspace-mode-docker-container"),
    );
    await user.click(screen.getByTestId("stub-chat-submit"));

    expect(screen.getByTestId("stub-workspace-modes")).toHaveTextContent(
      "local_repo,new_worktree,docker_container",
    );
    await waitFor(() =>
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({ workspaceMode: "docker_container" }),
      ),
    );
  });

  it("falls back from a remembered Docker mode the server does not offer", async () => {
    writeStoredLocalWorkspaceMode("docker_container");
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );

    expect(screen.getByTestId("stub-workspace-modes")).toHaveTextContent(
      "local_repo,new_worktree",
    );
    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:local_repo",
    );
  });

  it("starts on the Settings → Runtime default", async () => {
    writeStoredLocalWorkspaceMode("new_worktree");
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      conversation_settings: { default_runtime: "docker" },
    });
    mockUseConversationWorkspace.mockReturnValue({
      isolated: false,
      dockerSelectable: true,
      unsupportedMessage: null,
    });
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );

    await waitFor(() =>
      expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
        "local:docker_container",
      ),
    );
  });

  it("offers Local or Docker before a workspace is opened", async () => {
    mockUseConversationWorkspace.mockReturnValue({
      isolated: false,
      dockerSelectable: true,
      unsupportedMessage: null,
    });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-no-ws" }),
      );
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("workspace-mode-selector"));
    expect(
      screen
        .getAllByRole("menuitemradio")
        .map((option) => option.getAttribute("data-testid")),
    ).toEqual([
      "workspace-mode-selector-option-local_repo",
      "workspace-mode-selector-option-docker_container",
    ]);
    await user.click(
      screen.getByTestId("workspace-mode-selector-option-docker_container"),
    );
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() =>
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({ workspaceMode: "docker_container" }),
      ),
    );
    expect(createSpy.mock.calls[0][0]).not.toHaveProperty("workingDir");
  });

  it("keeps a remembered Docker mode while server_info is still loading", async () => {
    writeStoredLocalWorkspaceMode("docker_container");
    mockUseConversationWorkspace.mockReturnValue({
      isolated: false,
      dockerSelectable: false,
      serverInfoLoading: true,
      unsupportedMessage: null,
    });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(
        makeConversationResponse({ app_conversation_id: "conv-pending" }),
      );
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );
    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() =>
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({ workspaceMode: "docker_container" }),
      ),
    );
  });

  it("restores and updates the last selected local workspace mode", async () => {
    writeStoredLocalWorkspaceMode("new_worktree");
    const { unmount } = renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );

    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:new_worktree",
    );

    await user.click(screen.getByTestId("stub-workspace-mode-local-repo"));
    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:local_repo",
    );

    unmount();
    renderLauncher();

    await user.click(screen.getByTestId("open-workspace-button"));
    await user.click(
      await screen.findByTestId("stub-workspace-dialog-confirm"),
    );

    expect(screen.getByTestId("stub-workspace-mode")).toHaveTextContent(
      "local:local_repo",
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

  it("allows Automate mode on cloud backends", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());

    renderLauncher();
    const user = userEvent.setup();

    expect(
      screen.queryByText("HOME$AUTOMATE_LOCAL_BACKEND_ONLY"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByTestId("home-launcher-mode-automate"));

    expect(screen.getByTestId("home-launcher-mode-automate")).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.getByTestId("stub-chat-submit")).toHaveAttribute(
      "data-placeholder",
      "HOME$AUTOMATE_PROMPT_PLACEHOLDER",
    );

    await user.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({ automationSetup: true }),
    );
  });

  it("passes the picked repository + branch payload on a cloud backend", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    mockConnectedGitHub();
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
      expect.objectContaining({
        initialUserMsg: "hello world",
        metadata: {
          selected_repository: "org/repo",
          selected_branch: "main",
          git_provider: "github",
        },
      }),
    );
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-repo"),
    );
  });

  it("remembers the picked repository when the home page is revisited", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    mockConnectedGitHub();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());
    const user = userEvent.setup();
    const firstVisit = renderLauncher();
    await user.click(screen.getByTestId("open-repository-button"));
    await user.click(await screen.findByTestId("stub-repo-dialog-confirm"));
    firstVisit.unmount();

    renderLauncher();

    await waitFor(() =>
      expect(screen.getByTestId("stub-preview-repository")).toHaveTextContent(
        "org/repo",
      ),
    );
    expect(GitService.searchRepositoryBranches).toHaveBeenCalledWith(
      "org/repo",
      "github",
      "main",
      undefined,
      100,
    );
    await user.click(screen.getByTestId("stub-chat-submit"));
    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy.mock.lastCall?.[0]?.metadata).toEqual({
      selected_repository: "org/repo",
      selected_branch: "main",
      git_provider: "github",
    });
  });

  it.each([
    {
      description: "the repository can no longer be reached",
      arrange: () => {
        mockConnectedGitHub();
        vi.mocked(GitService.searchRepositoryBranches).mockRejectedValue(
          new Error("Not Found"),
        );
      },
    },
    {
      description: "the branch no longer exists",
      arrange: () => {
        mockConnectedGitHub();
        vi.mocked(GitService.searchRepositoryBranches).mockResolvedValue({
          items: [makeBranch("main-old")],
          next_page_id: null,
        });
      },
    },
    {
      description: "the provider is no longer connected",
      arrange: () => {
        vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
          MOCK_DEFAULT_USER_SETTINGS,
        );
      },
    },
  ])(
    "forgets a remembered repository when $description",
    async ({ arrange }) => {
      mockUseActiveBackend.mockReturnValue(cloudBackend);
      arrange();
      useHomeStore.getState().setRepositorySelection("cloud-id", {
        repository: {
          id: "1",
          full_name: "org/gone",
          git_provider: "github",
          is_public: false,
        },
        branch: makeBranch("main"),
        provider: "github",
      });
      const createSpy = vi
        .spyOn(AgentServerConversationService, "createConversation")
        .mockResolvedValue(makeConversationResponse());

      renderLauncher();

      expect(
        await screen.findByTestId("open-repository-button"),
      ).toBeInTheDocument();
      expect(useHomeStore.getState().repositorySelections).toEqual({});
      await userEvent.click(screen.getByTestId("stub-chat-submit"));
      await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
      expect(
        createSpy.mock.lastCall?.[0]?.metadata?.selected_repository,
      ).toBeUndefined();
    },
  );

  it("clears the repository selection so later visits start without one", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    mockConnectedGitHub();
    const user = userEvent.setup();
    const firstVisit = renderLauncher();
    await user.click(screen.getByTestId("open-repository-button"));
    await user.click(await screen.findByTestId("stub-repo-dialog-confirm"));

    await user.click(screen.getByTestId("stub-clear-selection"));

    expect(screen.getByTestId("open-repository-button")).toBeInTheDocument();
    firstVisit.unmount();
    renderLauncher();
    expect(screen.getByTestId("open-repository-button")).toBeInTheDocument();
  });

  it("keeps remembered repositories separate per cloud backend", async () => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    mockConnectedGitHub();
    const user = userEvent.setup();
    const firstVisit = renderLauncher();
    await user.click(screen.getByTestId("open-repository-button"));
    await user.click(await screen.findByTestId("stub-repo-dialog-confirm"));
    firstVisit.unmount();

    mockUseActiveBackend.mockReturnValue({
      ...cloudBackend,
      backend: { ...cloudBackend.backend, id: "other-cloud-id" },
    });
    renderLauncher();

    expect(screen.getByTestId("open-repository-button")).toBeInTheDocument();
  });

  it.each([
    {
      description: "forwards a member's home LLM pick",
      pick: "ops55",
      expected: "ops55",
    },
    {
      description: "drops a home LLM pick whose profile is gone",
      pick: "Deleted",
      expected: undefined,
    },
  ])("$description on a cloud launch", async ({ pick, expected }) => {
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    const listProfilesSpy = vi
      .spyOn(ProfilesService, "listProfiles")
      .mockResolvedValue({
        profiles: [
          {
            name: "ops55",
            model: "anthropic/claude-opus-5-5",
            base_url: null,
            api_key_set: true,
          },
        ],
        active_profile: null,
      });
    useModelStore.setState({ homeLlmProfile: pick });
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeConversationResponse());
    renderLauncher();
    await waitFor(() => expect(listProfilesSpy).toHaveBeenCalled());
    // Let the resolved profile list reach the launcher before submitting.
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));

    await userEvent.click(screen.getByTestId("stub-chat-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(createSpy.mock.lastCall?.[0]?.llmProfile).toBe(expected);
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
      expect.objectContaining({
        metadata: null,
      }),
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

  it("does not hand the prompt back to a composer when a step after a successful create fails", async () => {
    // Arrange — the conversation exists; only the follow-up enqueue fails.
    mockUseActiveBackend.mockReturnValue(cloudBackend);
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockResolvedValue(
      makeConversationResponse({
        id: "start-task-1",
        app_conversation_id: null,
      }),
    );
    enqueueHomeTaskPendingMessage.mockRejectedValue(new Error("Queue down"));

    // Act
    renderLauncher();
    await userEvent.setup().click(screen.getByTestId("stub-chat-submit"));

    // Assert — the prompt already belongs to the new conversation, so
    // restoring it would replay it into that conversation's composer.
    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("Queue down"),
    );
    expect(mockRestoreMessageToInputIfEmpty).not.toHaveBeenCalled();
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
      expect.objectContaining({
        metadata: null,
      }),
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

  it.each([
    {
      name: "code mode with picked plugins",
      prepare: async () => {
        await userEvent.click(screen.getByTestId("open-plugin-picker"));
        await userEvent.click(await screen.findByTestId("stub-plugin-pick"));
      },
      expectedPlugins: [{ source: "github:o/a", ref: null, repo_path: null }],
      expectedAutomationSetup: false,
      marksAutomationSetup: false,
    },
    {
      name: "automate mode",
      prepare: async () => {
        await userEvent.click(
          screen.getByTestId("home-launcher-mode-automate"),
        );
      },
      expectedPlugins: undefined,
      expectedAutomationSetup: true,
      marksAutomationSetup: true,
    },
  ])(
    "creates a conversation in $name",
    async ({
      prepare,
      expectedPlugins,
      expectedAutomationSetup,
      marksAutomationSetup,
    }) => {
      const createSpy = vi
        .spyOn(AgentServerConversationService, "createConversation")
        .mockResolvedValue(makeConversationResponse());

      renderLauncher();

      await prepare();
      await userEvent.click(screen.getByTestId("stub-chat-submit"));

      await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          initialUserMsg: "hello world",
          plugins: expectedPlugins,
          automationSetup: expectedAutomationSetup,
          metadata: null,
        }),
      );
      if (marksAutomationSetup) {
        // Marked once the create mutation settles, after the call above.
        await waitFor(() =>
          expect(mockMarkAutomationSetupHandoff).toHaveBeenCalledWith(
            "conv-abc",
          ),
        );
      } else {
        expect(mockMarkAutomationSetupHandoff).not.toHaveBeenCalled();
      }
    },
  );

  it("keeps Code mode focused on starter prompts instead of the recommended automations rail", async () => {
    const user = userEvent.setup();
    renderLauncher();

    expect(
      screen.queryByTestId("recommended-automations-rail"),
    ).not.toBeInTheDocument();
    const starterPrompts = screen.getByRole("group", {
      name: "HOME$STARTER_PROMPTS_LABEL",
    });
    const [label, prompt] = Object.entries(NON_REPO_SUGGESTIONS)[0];

    await user.click(
      within(starterPrompts).getByRole("button", { name: label }),
    );

    expect(mockSetMessageToSend).toHaveBeenCalledWith(prompt, "home");
  });

  it("swaps starter prompts for the recommended automations rail in Automate mode", async () => {
    renderLauncher();
    const user = userEvent.setup();

    await user.click(screen.getByTestId("home-launcher-mode-automate"));

    expect(
      screen.getByTestId("recommended-automations-rail"),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: "HOME$STARTER_PROMPTS_LABEL" }),
    ).not.toBeInTheDocument();
  });

  // @spec HL-002 — Dismissing Home's rail removes it and persists the choice.
  it("hides the Automate-mode recommended automations rail when dismissed", async () => {
    const user = userEvent.setup();
    renderLauncher();
    await user.click(screen.getByTestId("home-launcher-mode-automate"));

    const dismissBtn = screen.getByTestId(
      "recommended-automations-rail-dismiss",
    );
    await user.click(dismissBtn);

    expect(
      screen.queryByTestId("recommended-automations-rail"),
    ).not.toBeInTheDocument();
    expect(
      window.localStorage.getItem(
        HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
      ),
    ).toBe("true");
  });

  // @spec HL-002 — A stored dismissal keeps the rail off Home.
  it("does not render recommended automations rail when previously dismissed", async () => {
    window.localStorage.setItem(
      HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
      "true",
    );
    const user = userEvent.setup();

    renderLauncher();
    await user.click(screen.getByTestId("home-launcher-mode-automate"));

    expect(
      screen.queryByTestId("recommended-automations-rail"),
    ).not.toBeInTheDocument();
  });
});
