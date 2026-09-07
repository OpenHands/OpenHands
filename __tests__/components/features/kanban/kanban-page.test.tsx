import { screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import KanbanService from "#/api/kanban-service/kanban-service.api";
import { KANBAN_ALL_WORKSPACES_PATH } from "#/api/kanban-service/kanban-constants";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { writeKanbanWorkspacePath } from "#/components/features/kanban/kanban-workspace";
import { I18nKey } from "#/i18n/declaration";
import { resetKanbanMockData } from "#/mocks/handlers";
import KanbanPage from "#/routes/kanban";
import { useKanbanSyncStore } from "#/stores/kanban-sync-store";
import type { LocalWorkspace } from "#/types/workspace";

const ALPHA: LocalWorkspace = {
  id: "ws-alpha",
  name: "alpha",
  path: "/tmp/alpha",
};

const BETA: LocalWorkspace = {
  id: "ws-beta",
  name: "beta",
  path: "/tmp/beta",
};

const { mockSearchSubdirectories } = vi.hoisted(() => ({
  mockSearchSubdirectories: vi.fn(),
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
        getHome: vi.fn().mockResolvedValue({ home: "/tmp" }),
      };
    }),
  };
});

function mockWorkspaces(workspaces: LocalWorkspace[] = []) {
  vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
    workspaces,
    workspaceParents: [],
  });
}

describe("KanbanPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    resetKanbanMockData();
    window.sessionStorage.clear();
    window.localStorage.clear();
    useKanbanSyncStore.getState().reset();
    mockSearchSubdirectories.mockResolvedValue({ items: [] });
  });

  it("asks for a workspace instead of a free-floating board name", async () => {
    mockWorkspaces();
    renderWithProviders(<KanbanPage />);

    expect(await screen.findByTestId("kanban-empty")).toHaveTextContent(
      I18nKey.KANBAN$NO_WORKSPACE,
    );
    expect(screen.getByTestId("kanban-workspace-picker")).toBeInTheDocument();
    expect(screen.queryByTestId("kanban-board-name")).not.toBeInTheDocument();
    expect(screen.queryByTestId("kanban-create-board")).not.toBeInTheDocument();
  });

  it("creates a board bound to the selected workspace path", async () => {
    mockWorkspaces([ALPHA]);
    const createBoard = vi.spyOn(KanbanService, "createBoard");
    renderWithProviders(<KanbanPage />);

    await waitFor(() => {
      expect(createBoard).toHaveBeenCalledWith({
        name: "alpha",
        project_id: "/tmp/alpha",
      });
    });
    expect(await screen.findByTestId("kanban-board")).toBeInTheDocument();
  });

  it("opens the board whose project_id matches the workspace, not the first board", async () => {
    mockWorkspaces([ALPHA]);
    const other = await KanbanService.createBoard({
      name: "Other",
      project_id: "/tmp/other",
    });
    const matched = await KanbanService.createBoard({
      name: "Alpha board",
      project_id: "/tmp/alpha",
    });
    await KanbanService.createCard(other.columns[0].id, {
      title: "Other task",
    });
    await KanbanService.createCard(matched.columns[0].id, {
      title: "Alpha task",
    });

    renderWithProviders(<KanbanPage />);

    expect(await screen.findByText("Alpha task")).toBeInTheDocument();
    expect(screen.queryByText("Other task")).not.toBeInTheDocument();
  });

  it("switches to list view from the segmented control", async () => {
    mockWorkspaces([ALPHA]);
    const matched = await KanbanService.createBoard({
      name: "Alpha board",
      project_id: "/tmp/alpha",
    });
    const card = await KanbanService.createCard(matched.columns[0].id, {
      title: "Alpha task",
    });
    const user = userEvent.setup();
    renderWithProviders(<KanbanPage />);

    expect(await screen.findByTestId("kanban-board")).toBeInTheDocument();
    await user.click(screen.getByTestId("kanban-view-option-list"));
    expect(screen.getByTestId("kanban-list")).toBeInTheDocument();
    expect(screen.getByTestId(`kanban-list-row-${card.id}`)).toHaveTextContent(
      "Alpha task",
    );
  });

  it("shows a combined board with a swimlane per workspace", async () => {
    mockWorkspaces([ALPHA, BETA]);
    const alpha = await KanbanService.createBoard({
      name: "Alpha board",
      project_id: "/tmp/alpha",
    });
    const beta = await KanbanService.createBoard({
      name: "Beta board",
      project_id: "/tmp/beta",
    });
    await KanbanService.createCard(alpha.columns[0].id, {
      title: "Alpha task",
    });
    await KanbanService.createCard(beta.columns[0].id, { title: "Beta task" });
    writeKanbanWorkspacePath(KANBAN_ALL_WORKSPACES_PATH);

    renderWithProviders(<KanbanPage />);

    expect(
      await screen.findByTestId(`kanban-swimlane-workspace:${alpha.id}`),
    ).toHaveTextContent("alpha");
    expect(
      screen.getByTestId(`kanban-swimlane-workspace:${beta.id}`),
    ).toHaveTextContent("beta");
    expect(screen.getByText("Alpha task")).toBeInTheDocument();
    expect(screen.getByText("Beta task")).toBeInTheDocument();
  });

  it("maps a source and opens reconcile with a recommendation", async () => {
    mockWorkspaces([ALPHA]);
    const matched = await KanbanService.createBoard({
      name: "Alpha board",
      project_id: "/tmp/alpha",
    });
    await KanbanService.createCard(matched.columns[0].id, {
      title: "Ship login",
    });
    const user = userEvent.setup();
    renderWithProviders(<KanbanPage />);

    expect(await screen.findByTestId("kanban-board")).toBeInTheDocument();
    await user.click(screen.getByTestId("kanban-map-source"));
    expect(screen.getByTestId("kanban-map-source-modal")).toBeInTheDocument();
    await user.type(screen.getByTestId("kanban-source-label"), "Linear");
    fireEvent.change(screen.getByTestId("kanban-source-issues"), {
      target: {
        value: JSON.stringify([
          {
            id: "LIN-1",
            title: "Ship login",
            description: "Remote copy",
            group: "Auth",
          },
        ]),
      },
    });
    await user.click(screen.getByTestId("kanban-map-source-submit"));

    expect(
      await screen.findByTestId("kanban-reconcile-modal"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("kanban-conflict-banner")).toBeInTheDocument();
    expect(screen.getByText("Auth")).toBeInTheDocument();
  });
});
