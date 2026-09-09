import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import KanbanService from "#/api/kanban-service/kanban-service.api";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { I18nKey } from "#/i18n/declaration";
import { resetKanbanMockData } from "#/mocks/handlers";
import KanbanPage from "#/routes/kanban";
import type { LocalWorkspace } from "#/types/workspace";

const ALPHA: LocalWorkspace = {
  id: "ws-alpha",
  name: "alpha",
  path: "/tmp/alpha",
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
});
