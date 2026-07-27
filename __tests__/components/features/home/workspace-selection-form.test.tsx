import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, vi, beforeEach, it } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import {
  HOME_SELECTED_WORKSPACE_PATH_KEY,
  WorkspaceSelectionForm,
} from "../../../../src/components/features/home/workspace-selection-form";
import { WorkspaceDropdown } from "../../../../src/components/features/home/workspace-dropdown/workspace-dropdown";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";

const mockNavigate = vi.fn();
const mockUseIsCreatingConversation = vi.fn();

const { mockSearchSubdirectories, mockGetHome } = vi.hoisted(() => ({
  mockSearchSubdirectories: vi.fn(),
  mockGetHome: vi.fn(),
}));

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

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => mockUseIsCreatingConversation(),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationCreated: vi.fn(),
    trackLoginButtonClick: vi.fn(),
  }),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    FileClient: vi.fn(function FileClientMock() {
      return {
        searchSubdirectories: mockSearchSubdirectories,
        getHome: mockGetHome,
      };
    }),
  };
});

mockUseIsCreatingConversation.mockReturnValue(false);

function renderForm({
  workspaces = [],
  workspaceParents = [],
}: {
  workspaces?: LocalWorkspace[];
  workspaceParents?: LocalWorkspaceParent[];
} = {}) {
  vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
    workspaces,
    workspaceParents,
  });
  return render(<WorkspaceSelectionForm />, {
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
}

async function openWorkspaceDropdown(user: ReturnType<typeof userEvent.setup>) {
  const dropdown = await screen.findByTestId("workspace-dropdown");
  await waitFor(() => expect(dropdown).not.toBeDisabled());
  await user.click(dropdown);
  return screen.findByTestId("workspace-dropdown-menu");
}

describe("WorkspaceDropdown", () => {
  it.each([
    {
      testId: "add-workspaces-button",
      expectedCallback: "add",
    },
    {
      testId: "manage-workspaces-button",
      expectedCallback: "manage",
    },
  ])(
    "opens $expectedCallback workspace action from touch without bubbling",
    async ({ testId, expectedCallback }) => {
      const outsideTouchEnd = vi.fn();
      const onAddClick = vi.fn();
      const onManageClick = vi.fn();
      const user = userEvent.setup();

      render(
        <div onTouchEnd={outsideTouchEnd}>
          <WorkspaceDropdown
            workspaces={[
              {
                id: "/Users/me/dev/repo1",
                name: "repo1",
                path: "/Users/me/dev/repo1",
              },
            ]}
            value={null}
            onChange={vi.fn()}
            onAddClick={onAddClick}
            onManageClick={onManageClick}
          />
        </div>,
      );

      await user.click(await screen.findByTestId("workspace-dropdown"));
      const action = await screen.findByTestId(testId);

      fireEvent.touchStart(action);
      fireEvent.touchEnd(action);

      expect(outsideTouchEnd).not.toHaveBeenCalled();
      expect(onAddClick).toHaveBeenCalledTimes(
        expectedCallback === "add" ? 1 : 0,
      );
      expect(onManageClick).toHaveBeenCalledTimes(
        expectedCallback === "manage" ? 1 : 0,
      );
    },
  );
});

describe("WorkspaceSelectionForm (server-backed workspaces)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockNavigate.mockReset();
    mockSearchSubdirectories.mockReset();
    mockGetHome.mockReset();
    window.sessionStorage.clear();
    mockUseIsCreatingConversation.mockReturnValue(false);
    mockGetHome.mockResolvedValue({ home: "/Users/me" });
    // useResolvedWorkspaces always queries an implicit `/projects` parent in
    // dev mode — return empty so it doesn't influence tests that don't care.
    mockSearchSubdirectories.mockResolvedValue({
      items: [],
      next_page_id: null,
    });
  });

  it("renders the default empty selection when no workspace path is persisted", async () => {
    renderForm({
      workspaces: [
        {
          id: "/Users/me/dev/repo1",
          name: "repo1",
          path: "/Users/me/dev/repo1",
        },
      ],
    });

    const dropdown = await screen.findByTestId("workspace-dropdown");
    await waitFor(() => expect(dropdown).not.toBeDisabled());

    expect(dropdown).toHaveValue("");
    expect(screen.getByTestId("workspace-launch-button")).toBeDisabled();
    expect(
      window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
    ).toBeNull();
  });

  it("renders workspaces returned by the agent-server in the dropdown", async () => {
    // Arrange
    renderForm({
      workspaces: [
        {
          id: "/Users/me/dev/repo1",
          name: "repo1",
          path: "/Users/me/dev/repo1",
        },
      ],
    });
    const user = userEvent.setup();

    // Act
    const menu = await openWorkspaceDropdown(user);

    // Assert
    expect(await within(menu).findByText("repo1")).toBeInTheDocument();
  });

  it("renders parent-derived workspaces from all paginated folders", async () => {
    const workspaceParent = {
      id: "/Users/me/dev",
      name: "dev",
      path: "/Users/me/dev",
    };
    mockSearchSubdirectories.mockImplementation(
      async (path: string, options?: { pageId?: string | null }) => {
        if (path === workspaceParent.path && !options?.pageId) {
          return {
            items: [{ name: "alpha", path: "/Users/me/dev/alpha" }],
            next_page_id: "page-2",
          };
        }
        if (path === workspaceParent.path && options?.pageId === "page-2") {
          return {
            items: [{ name: "paper", path: "/Users/me/dev/paper" }],
            next_page_id: null,
          };
        }
        return { items: [], next_page_id: null };
      },
    );
    renderForm({ workspaceParents: [workspaceParent] });
    const user = userEvent.setup();

    const menu = await openWorkspaceDropdown(user);

    expect(await within(menu).findByText("alpha")).toBeInTheDocument();
    expect(await within(menu).findByText("paper")).toBeInTheDocument();
    expect(mockSearchSubdirectories).toHaveBeenCalledWith(
      workspaceParent.path,
      undefined,
    );
    expect(mockSearchSubdirectories).toHaveBeenCalledWith(
      workspaceParent.path,
      {
        pageId: "page-2",
      },
    );
  });

  it("restores the selected workspace after unmounting and remounting", async () => {
    const workspace = {
      id: "/Users/me/dev/repo1",
      name: "repo1",
      path: "/Users/me/dev/repo1",
    };
    const user = userEvent.setup();

    const { unmount } = renderForm({ workspaces: [workspace] });
    const firstMenu = await openWorkspaceDropdown(user);
    await user.click(await within(firstMenu).findByText("repo1"));

    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
      ).toBe(workspace.path),
    );
    expect(screen.getByTestId("workspace-launch-button")).not.toBeDisabled();

    unmount();
    renderForm({ workspaces: [workspace] });

    await waitFor(() =>
      expect(screen.getByTestId("workspace-dropdown")).toHaveValue("repo1"),
    );
    expect(screen.getByTestId("workspace-launch-button")).not.toBeDisabled();
  });

  it("clears a persisted workspace path that is no longer resolved", async () => {
    window.sessionStorage.setItem(
      HOME_SELECTED_WORKSPACE_PATH_KEY,
      "/Users/me/dev/missing",
    );

    renderForm({
      workspaces: [
        {
          id: "/Users/me/dev/repo1",
          name: "repo1",
          path: "/Users/me/dev/repo1",
        },
      ],
    });

    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
      ).toBeNull(),
    );
    expect(screen.getByTestId("workspace-dropdown")).toHaveValue("");
    expect(screen.getByTestId("workspace-launch-button")).toBeDisabled();
  });

  it("shows a version-specific workspace message for old agent servers", async () => {
    vi.spyOn(WorkspacesService, "listWorkspaces").mockRejectedValue({
      code: "AGENT_SERVER_VERSION_TOO_OLD",
      feature: "workspaces",
      requiredVersion: "1.23.0",
      actualVersion: "1.22.1",
    });

    render(<WorkspaceSelectionForm />, {
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

    await waitFor(() =>
      expect(screen.getByTestId("workspace-dropdown")).toBeDisabled(),
    );
    await waitFor(() =>
      expect(screen.getByTestId("workspace-dropdown")).toHaveAttribute(
        "placeholder",
        "HOME$WORKSPACES_UNSUPPORTED_PLACEHOLDER",
      ),
    );
    expect(screen.getByTestId("workspace-status-message")).toHaveTextContent(
      "HOME$WORKSPACES_UNSUPPORTED_AGENT_SERVER",
    );
  });

  it("Add Workspace dispatches addWorkspaces to the agent-server", async () => {
    // Arrange
    const addSpy = vi
      .spyOn(WorkspacesService, "addWorkspaces")
      .mockResolvedValue({ workspaces: [], workspaceParents: [] });
    mockSearchSubdirectories.mockImplementation(async (path: string) => {
      if (path === "/Users/me") {
        return {
          items: [{ name: "dev", path: "/Users/me/dev" }],
          next_page_id: null,
        };
      }
      return { items: [], next_page_id: null };
    });
    renderForm();
    const user = userEvent.setup();

    // Act
    await user.click(await screen.findByTestId("workspace-dropdown"));
    await user.click(await screen.findByTestId("add-workspaces-button"));
    await screen.findByTestId("folder-browser-modal");
    await user.click(await screen.findByTestId("folder-browser-entry-dev"));
    await user.click(screen.getByTestId("folder-browser-use"));

    // Assert
    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    expect(addSpy).toHaveBeenCalledWith([
      { id: "/Users/me/dev", name: "dev", path: "/Users/me/dev" },
    ]);
  });

  it("shows all paginated folders in the Add Workspace browser", async () => {
    mockSearchSubdirectories.mockImplementation(
      async (path: string, options?: { pageId?: string | null }) => {
        if (path === "/Users/me" && !options?.pageId) {
          return {
            items: [{ name: "alpha", path: "/Users/me/alpha" }],
            next_page_id: "page-2",
          };
        }
        if (path === "/Users/me" && options?.pageId === "page-2") {
          return {
            items: [
              { name: "paper", path: "/Users/me/paper" },
              { name: "zeta", path: "/Users/me/zeta" },
            ],
            next_page_id: null,
          };
        }
        return { items: [], next_page_id: null };
      },
    );
    renderForm();
    const user = userEvent.setup();

    await user.click(await screen.findByTestId("workspace-dropdown"));
    await user.click(await screen.findByTestId("add-workspaces-button"));
    await screen.findByTestId("folder-browser-modal");

    expect(
      await screen.findByTestId("folder-browser-entry-alpha"),
    ).toBeVisible();
    expect(
      await screen.findByTestId("folder-browser-entry-paper"),
    ).toBeVisible();
    expect(
      await screen.findByTestId("folder-browser-entry-zeta"),
    ).toBeVisible();
    expect(mockSearchSubdirectories).toHaveBeenCalledWith(
      "/Users/me",
      undefined,
    );
    expect(mockSearchSubdirectories).toHaveBeenCalledWith("/Users/me", {
      pageId: "page-2",
    });
  });

  it("handles Windows paths when browsing and adding a workspace", async () => {
    const homePath = String.raw`C:\Users\me`;
    const devPath = String.raw`C:\Users\me\dev`;
    const addSpy = vi
      .spyOn(WorkspacesService, "addWorkspaces")
      .mockResolvedValue({ workspaces: [], workspaceParents: [] });
    mockGetHome.mockResolvedValue({ home: homePath });
    mockSearchSubdirectories.mockImplementation(async (dir: string) => {
      if (dir === homePath) {
        return {
          items: [{ name: "dev", path: devPath }],
          next_page_id: null,
        };
      }
      return { items: [], next_page_id: null };
    });
    renderForm();
    const user = userEvent.setup();

    await user.click(await screen.findByTestId("workspace-dropdown"));
    await user.click(await screen.findByTestId("add-workspaces-button"));
    await screen.findByTestId("folder-browser-modal");
    await expect(
      screen.getByTestId("folder-browser-current-path"),
    ).toHaveTextContent(homePath);

    await user.click(await screen.findByTestId("folder-browser-entry-dev"));
    await expect(
      screen.getByTestId("folder-browser-current-path"),
    ).toHaveTextContent(devPath);

    await user.click(screen.getByTestId("folder-browser-up"));
    await expect(
      screen.getByTestId("folder-browser-current-path"),
    ).toHaveTextContent(homePath);

    await user.click(await screen.findByTestId("folder-browser-entry-dev"));
    await user.click(screen.getByTestId("folder-browser-use"));

    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    expect(addSpy).toHaveBeenCalledWith([
      { id: devPath, name: "dev", path: devPath },
    ]);
  });

  it("auto-selects the newly added workspace after Add Workspace", async () => {
    // Arrange: start with another workspace already selected, mirroring the
    // repro in OpenHands/agent-canvas#1212.
    const existingWorkspace = {
      id: "/Users/me/dev/repo1",
      name: "repo1",
      path: "/Users/me/dev/repo1",
    };
    const addedWorkspace = {
      id: "/Users/me/dev",
      name: "dev",
      path: "/Users/me/dev",
    };
    const addSpy = vi
      .spyOn(WorkspacesService, "addWorkspaces")
      .mockImplementation(async (items) => {
        // Mimic the server: once added, the list endpoint includes the new
        // workspace so the post-add refetch resolves it.
        const workspaces = [existingWorkspace, ...items];
        vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
          workspaces,
          workspaceParents: [],
        });
        return { workspaces, workspaceParents: [] };
      });
    mockSearchSubdirectories.mockImplementation(async (path: string) => {
      if (path === "/Users/me") {
        return {
          items: [{ name: "dev", path: "/Users/me/dev" }],
          next_page_id: null,
        };
      }
      return { items: [], next_page_id: null };
    });
    renderForm({ workspaces: [existingWorkspace] });
    const user = userEvent.setup();

    const selectionMenu = await openWorkspaceDropdown(user);
    await user.click(await within(selectionMenu).findByText("repo1"));
    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
      ).toBe(existingWorkspace.path),
    );

    // Act
    await openWorkspaceDropdown(user);
    await user.click(await screen.findByTestId("add-workspaces-button"));
    await screen.findByTestId("folder-browser-modal");
    await user.click(await screen.findByTestId("folder-browser-entry-dev"));
    await user.click(screen.getByTestId("folder-browser-use"));

    // Assert
    await waitFor(() => expect(addSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getByTestId("workspace-dropdown")).toHaveValue(
        addedWorkspace.name,
      ),
    );
    expect(
      window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
    ).toBe(addedWorkspace.path);
    expect(screen.getByTestId("workspace-launch-button")).not.toBeDisabled();
  });

  it("Remove Workspace dispatches removeWorkspace and clears the selected workspace", async () => {
    // Arrange
    const removeSpy = vi
      .spyOn(WorkspacesService, "removeWorkspace")
      .mockResolvedValue();
    const workspace = {
      id: "/Users/me/dev/repo1",
      name: "repo1",
      path: "/Users/me/dev/repo1",
    };
    renderForm({
      workspaces: [workspace],
    });
    const user = userEvent.setup();

    // Act
    const selectionMenu = await openWorkspaceDropdown(user);
    await user.click(await within(selectionMenu).findByText("repo1"));
    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
      ).toBe(workspace.path),
    );

    await openWorkspaceDropdown(user);
    await user.click(await screen.findByTestId("manage-workspaces-button"));
    await screen.findByTestId("manage-workspaces-modal");
    await user.click(screen.getByTestId("manage-workspaces-remove-repo1"));
    await screen.findByTestId("confirmation-modal");
    await user.click(screen.getByTestId("confirm-button"));

    // Assert
    await waitFor(() => expect(removeSpy).toHaveBeenCalledTimes(1));
    expect(removeSpy).toHaveBeenCalledWith("/Users/me/dev/repo1");
    expect(
      window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
    ).toBeNull();
    expect(screen.getByTestId("workspace-dropdown")).toHaveValue("");
    expect(screen.getByTestId("workspace-launch-button")).toBeDisabled();
  });

  it("clears the selected parent-derived workspace when its parent is removed", async () => {
    const removeParentSpy = vi
      .spyOn(WorkspacesService, "removeWorkspaceParent")
      .mockResolvedValue();
    const workspaceParent = {
      id: "/Users/me/dev",
      name: "dev",
      path: "/Users/me/dev",
    };
    const workspacePath = "/Users/me/dev/repo1";
    mockSearchSubdirectories.mockImplementation(async (path: string) => {
      if (path === workspaceParent.path) {
        return {
          items: [{ name: "repo1", path: workspacePath }],
          next_page_id: null,
        };
      }
      return { items: [], next_page_id: null };
    });
    renderForm({ workspaceParents: [workspaceParent] });
    const user = userEvent.setup();

    const selectionMenu = await openWorkspaceDropdown(user);
    await user.click(await within(selectionMenu).findByText("repo1"));
    await waitFor(() =>
      expect(
        window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
      ).toBe(workspacePath),
    );

    await openWorkspaceDropdown(user);
    await user.click(await screen.findByTestId("manage-workspaces-button"));
    await screen.findByTestId("manage-workspaces-modal");
    await user.click(screen.getByTestId("manage-workspaces-remove-parent-dev"));
    await screen.findByTestId("confirmation-modal");
    await user.click(screen.getByTestId("confirm-button"));

    await waitFor(() => expect(removeParentSpy).toHaveBeenCalledTimes(1));
    expect(removeParentSpy).toHaveBeenCalledWith(workspaceParent.path);
    expect(
      window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
    ).toBeNull();
    expect(screen.getByTestId("workspace-dropdown")).toHaveValue("");
    expect(screen.getByTestId("workspace-launch-button")).toBeDisabled();
  });

  it("Add all subdirectories dispatches addWorkspaceParents to the agent-server", async () => {
    // Arrange
    const addParentsSpy = vi
      .spyOn(WorkspacesService, "addWorkspaceParents")
      .mockResolvedValue({ workspaces: [], workspaceParents: [] });
    mockSearchSubdirectories.mockImplementation(async (path: string) => {
      if (path === "/Users/me") {
        return {
          items: [{ name: "dev", path: "/Users/me/dev" }],
          next_page_id: null,
        };
      }
      return { items: [], next_page_id: null };
    });
    renderForm();
    const user = userEvent.setup();

    // Act
    await user.click(await screen.findByTestId("workspace-dropdown"));
    await user.click(await screen.findByTestId("add-workspaces-button"));
    await screen.findByTestId("folder-browser-modal");
    await user.click(await screen.findByTestId("folder-browser-entry-dev"));
    await user.click(screen.getByTestId("folder-browser-add-all-subdirs"));

    // Assert
    await waitFor(() => expect(addParentsSpy).toHaveBeenCalledTimes(1));
    expect(addParentsSpy).toHaveBeenCalledWith([
      { id: "/Users/me/dev", name: "dev", path: "/Users/me/dev" },
    ]);
  });
});
