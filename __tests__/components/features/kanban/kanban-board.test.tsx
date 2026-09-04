import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { KANBAN_DONE_STATUS } from "#/api/kanban-service/kanban-constants";
import type {
  KanbanBoard,
  KanbanBoardCosts,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import { CardDetailPanel } from "#/components/features/kanban/card-detail-panel";
import { CostSummary } from "#/components/features/kanban/cost-summary";
import { KanbanBoardView } from "#/components/features/kanban/kanban-board";
import { KanbanCard as KanbanCardView } from "#/components/features/kanban/kanban-card";
import { KanbanList } from "#/components/features/kanban/kanban-list";
import { HOME_SELECTED_WORKSPACE_PATH_KEY } from "#/components/features/home/workspace-selection-form";
import {
  boardForWorkspace,
  KANBAN_SELECTED_WORKSPACE_PATH_KEY,
  writeKanbanWorkspacePath,
} from "#/components/features/kanban/kanban-workspace";

function makeCard(overrides: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id: "card-1",
    column_id: "col-1",
    board_id: "board-1",
    title: "Ship API",
    description: "CRUD endpoints",
    priority: "P0",
    status: "todo",
    assignee: "agent",
    linked_branch: "feat/kanban",
    linked_pr: "https://example.com/pr/1",
    estimate_tokens: 1000,
    estimate_cost: 1.25,
    actual_tokens: null,
    actual_cost: null,
    model_used: null,
    tool_calls: null,
    agent_time: null,
    agent_session_id: null,
    position: 0,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

function makeBoard(cards: KanbanCard[] = [makeCard()]): KanbanBoard {
  return {
    id: "board-1",
    name: "Work",
    project_id: null,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    columns: [
      {
        id: "col-1",
        board_id: "board-1",
        name: "Backlog",
        position: 0,
        color: "#6b7280",
        cards,
      },
      {
        id: "col-2",
        board_id: "board-1",
        name: "Review",
        position: 1,
        color: "#f59e0b",
        cards: [],
      },
    ],
  };
}

const costs: KanbanBoardCosts = {
  board_id: "board-1",
  total_estimate_cost: 1.25,
  total_actual_cost: 0,
  total_estimate_tokens: 1000,
  total_actual_tokens: 0,
  columns: [
    {
      id: "col-1",
      name: "Backlog",
      estimate_cost: 1.25,
      actual_cost: 0,
      estimate_tokens: 1000,
      actual_tokens: 0,
    },
    {
      id: "col-2",
      name: "Review",
      estimate_cost: 0,
      actual_cost: 0,
      estimate_tokens: 0,
      actual_tokens: 0,
    },
  ],
};

describe("KanbanCard", () => {
  it("shows the estimate while the card is not done", () => {
    renderWithProviders(<KanbanCardView card={makeCard()} />);
    expect(screen.getByTestId("kanban-card-card-1")).toHaveTextContent(
      "Ship API",
    );
    expect(screen.getByTestId("kanban-card-priority-card-1")).toHaveTextContent(
      "P0",
    );
    expect(
      screen.getByTestId("kanban-card-cost-kind-card-1"),
    ).toHaveTextContent("KANBAN$COST_ESTIMATE");
    expect(screen.getByTestId("kanban-card-cost-card-1")).toHaveTextContent(
      "$1.25",
    );
  });

  it("shows actual cost when the card is done", () => {
    renderWithProviders(
      <KanbanCardView
        card={makeCard({
          status: KANBAN_DONE_STATUS,
          actual_cost: 0.9,
        })}
      />,
    );
    expect(
      screen.getByTestId("kanban-card-cost-kind-card-1"),
    ).toHaveTextContent("KANBAN$COST_ACTUAL");
    expect(screen.getByTestId("kanban-card-cost-card-1")).toHaveTextContent(
      "$0.90",
    );
  });
});

describe("KanbanBoardView", () => {
  it("shows column aggregate cost and moves a card on drop", () => {
    const onMoveCard = vi.fn();
    renderWithProviders(
      <KanbanBoardView
        board={makeBoard()}
        costs={costs}
        onMoveCard={onMoveCard}
      />,
    );
    expect(screen.getByTestId("kanban-column-cost-col-1")).toHaveTextContent(
      "$1.25",
    );
    const review = screen.getByTestId("kanban-column-col-2");
    fireEvent.drop(review, {
      dataTransfer: { getData: () => "card-1" },
    });
    expect(onMoveCard).toHaveBeenCalledWith("card-1", "col-2", 0);
  });
});

describe("CostSummary", () => {
  it("shows the board total", () => {
    renderWithProviders(<CostSummary costs={costs} />);
    expect(screen.getByTestId("kanban-board-total-cost")).toHaveTextContent(
      "$1.25",
    );
  });
});

describe("CardDetailPanel", () => {
  it("shows description, cost breakdown, and linked git refs", () => {
    renderWithProviders(
      <CardDetailPanel card={makeCard()} onClose={vi.fn()} />,
    );
    expect(screen.getByTestId("kanban-card-description")).toHaveValue(
      "CRUD endpoints",
    );
    expect(screen.getByTestId("kanban-detail-estimate")).toHaveTextContent(
      "$1.25",
    );
    expect(screen.getByTestId("kanban-detail-branch")).toHaveTextContent(
      "feat/kanban",
    );
    expect(screen.getByTestId("kanban-detail-pr")).toHaveTextContent(
      "https://example.com/pr/1",
    );
  });

  it("renders activity log entries when present", () => {
    renderWithProviders(
      <CardDetailPanel
        card={makeCard({
          activity_log: [
            { timestamp: "2026-09-01T00:00:00Z", message: "Linked session" },
          ],
        })}
        onClose={vi.fn()}
      />,
    );
    expect(screen.getByTestId("kanban-activity-log")).toHaveTextContent(
      "Linked session",
    );
  });
});

describe("boardForWorkspace", () => {
  it("returns the board whose project_id matches the workspace path", () => {
    const boards = [
      { ...makeBoard(), id: "other", project_id: "/tmp/other" },
      { ...makeBoard(), id: "mine", project_id: "/tmp/openhands" },
    ];
    expect(boardForWorkspace(boards, "/tmp/openhands")?.id).toBe("mine");
  });

  it("returns null when no workspace is selected", () => {
    expect(boardForWorkspace([makeBoard()], null)).toBeNull();
  });

  it("writes the kanban and home workspace keys together", () => {
    writeKanbanWorkspacePath("/tmp/openhands");
    expect(
      window.sessionStorage.getItem(KANBAN_SELECTED_WORKSPACE_PATH_KEY),
    ).toBe("/tmp/openhands");
    expect(
      window.sessionStorage.getItem(HOME_SELECTED_WORKSPACE_PATH_KEY),
    ).toBe("/tmp/openhands");
  });
});

describe("KanbanList", () => {
  it("sorts by created timestamp when that header is clicked", () => {
    const older = makeCard({
      id: "older",
      title: "Older",
      created_at: "2026-08-01T00:00:00Z",
      priority: "P3",
    });
    const newer = makeCard({
      id: "newer",
      title: "Newer",
      created_at: "2026-09-02T00:00:00Z",
      priority: "P0",
    });
    renderWithProviders(<KanbanList board={makeBoard([newer, older])} />);
    fireEvent.click(screen.getByTestId("kanban-list-sort-created"));
    const rows = screen.getAllByTestId(/kanban-list-row-/);
    expect(rows[0]).toHaveAttribute("data-testid", "kanban-list-row-older");
    expect(rows[1]).toHaveAttribute("data-testid", "kanban-list-row-newer");
  });
});
