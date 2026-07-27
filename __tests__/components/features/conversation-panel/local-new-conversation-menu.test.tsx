import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "test-utils";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { LocalNewConversationMenu } from "#/components/features/conversation-panel/local-new-conversation-menu";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

const { mockSearchSubdirectories } = vi.hoisted(() => ({
  mockSearchSubdirectories: vi.fn(),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
  }),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    FileClient: vi.fn(function FileClientMock() {
      return { searchSubdirectories: mockSearchSubdirectories };
    }),
  };
});

vi.mock(
  "#/components/features/home/workspace-dropdown/folder-browser-modal",
  () => ({
    FolderBrowserModal: ({ isOpen }: { isOpen: boolean }) =>
      isOpen ? <div data-testid="folder-browser-modal" /> : null,
  }),
);

vi.mock(
  "#/components/features/home/workspace-dropdown/manage-workspaces-modal",
  () => ({
    ManageWorkspacesModal: ({ isOpen }: { isOpen: boolean }) =>
      isOpen ? <div data-testid="manage-workspaces-modal" /> : null,
  }),
);

const makeStartTask = (conversationId: string) => ({
  id: "task-id",
  created_by_user_id: null,
  status: "READY" as const,
  detail: null,
  app_conversation_id: conversationId,
  agent_server_url: "http://agent-server.local",
  request: {
    initial_message: null,
    processors: [],
    llm_model: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: "github" as const,
    suggested_task: null,
    title: null,
    trigger: null,
    pr_number: [],
    parent_conversation_id: null,
    agent_type: "default" as const,
  },
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
});

const renderMenu = ({
  workspaces = [],
  workspaceParents = [],
  navigate,
}: {
  workspaces?: LocalWorkspace[];
  workspaceParents?: LocalWorkspaceParent[];
  navigate?: (to: string) => void;
} = {}) => {
  vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
    workspaces,
    workspaceParents,
  });
  return renderWithProviders(
    <LocalNewConversationMenu
      popoverClassName="left-0 right-0"
      trigger={(tp) => (
        <button type="button" data-testid="new-conversation-button" {...tp}>
          + New conversation
        </button>
      )}
    />,
    { navigation: { navigate, currentPath: "/conversations" } },
  );
};

describe("LocalNewConversationMenu", () => {
  beforeEach(() => {
    mockSearchSubdirectories.mockReset();
    mockSearchSubdirectories.mockResolvedValue({ items: [], next_page_id: null });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a divider above the workspace footer actions", async () => {
    const user = userEvent.setup();
    renderMenu();

    await user.click(screen.getByTestId("new-conversation-button"));

    expect(
      screen.getByTestId("new-conversation-menu-footer-divider"),
    ).toBeInTheDocument();
  });

  it("toggles the popover and dismisses it on outside click", async () => {
    // Arrange
    renderMenu();
    const user = userEvent.setup();

    // Act + Assert: open
    await user.click(screen.getByTestId("new-conversation-button"));
    expect(screen.getByTestId("new-conversation-popover")).toBeInTheDocument();

    // Act + Assert: outside click closes
    await user.click(document.body);
    await waitFor(() => {
      expect(
        screen.queryByTestId("new-conversation-popover"),
      ).not.toBeInTheDocument();
    });
  });

  it("dismisses the popover on Escape", async () => {
    // Arrange
    renderMenu();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("new-conversation-button"));
    expect(screen.getByTestId("new-conversation-popover")).toBeInTheDocument();

    // Act
    await user.keyboard("{Escape}");

    // Assert
    await waitFor(() => {
      expect(
        screen.queryByTestId("new-conversation-popover"),
      ).not.toBeInTheDocument();
    });
  });

  it("launches a conversation for the selected workspace", async () => {
    // Arrange
    const navigate = vi.fn();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockResolvedValue(makeStartTask("conv-123"));
    renderMenu({
      workspaces: [
        {
          id: "/workspace/project/repo1",
          name: "repo1",
          path: "/workspace/project/repo1",
        },
      ],
      navigate,
    });
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("new-conversation-button"));
    await user.click(await screen.findByRole("button", { name: "repo1" }));

    // Assert
    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        undefined,
        undefined,
        undefined,
        null,
        "/workspace/project/repo1",
        undefined,
        undefined,
        undefined,
      );
    });
    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith("/conversations/conv-123");
    });
  });

  it("disables launch actions while a conversation is being created", async () => {
    // Arrange
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockImplementation(() => new Promise(() => {}));
    renderMenu({
      workspaces: [
        {
          id: "/workspace/project/repo1",
          name: "repo1",
          path: "/workspace/project/repo1",
        },
      ],
    });
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("new-conversation-button"));
    await user.click(screen.getByTestId("launch-no-workspace"));

    // Assert
    await waitFor(() => {
      expect(screen.getByTestId("launch-no-workspace")).toBeDisabled();
      expect(screen.getByTestId("launch-workspace")).toBeDisabled();
    });
  });

  it("keeps the popover open while a workspace modal is open", async () => {
    // Arrange — add-modal exercises the shared `keepPopoverOpenOnMouseDown`
    // handler; manage-modal uses the same handler so we don't duplicate.
    renderMenu();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("new-conversation-button"));

    // Act
    await user.click(screen.getByTestId("add-workspaces-button"));

    // Assert
    expect(screen.getByTestId("folder-browser-modal")).toBeInTheDocument();
    await user.click(document.body);
    expect(screen.getByTestId("new-conversation-popover")).toBeInTheDocument();
  });

  it("keeps the popover open when conversation creation fails", async () => {
    // Arrange
    const navigate = vi.fn();
    const createSpy = vi
      .spyOn(AgentServerConversationService, "createConversation")
      .mockRejectedValue(new Error("create failed"));
    renderMenu({ navigate });
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("new-conversation-button"));
    await user.click(screen.getByTestId("launch-no-workspace"));

    // Assert
    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1);
    });
    await waitFor(() => {
      expect(screen.getByTestId("launch-no-workspace")).not.toBeDisabled();
    });
    expect(navigate).not.toHaveBeenCalled();
    expect(screen.getByTestId("new-conversation-popover")).toBeInTheDocument();
  });
});
