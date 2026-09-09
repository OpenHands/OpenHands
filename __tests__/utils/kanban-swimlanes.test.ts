import { describe, expect, it } from "vitest";
import type {
  KanbanBoard,
  KanbanCard,
} from "#/api/kanban-service/kanban-types";
import {
  alignBoardColumns,
  buildBoardSwimlanes,
  buildWorkspaceSwimlanes,
  laneCards,
  UNGROUPED_LANE_ID,
} from "#/utils/kanban-swimlanes";
import type { KanbanLane } from "#/utils/kanban-sync";

function makeCard(overrides: Partial<KanbanCard> = {}): KanbanCard {
  return {
    id: "card-1",
    column_id: "col-backlog",
    board_id: "board-alpha",
    title: "Alpha task",
    description: null,
    priority: "P2",
    status: "todo",
    assignee: null,
    linked_branch: null,
    linked_pr: null,
    estimate_tokens: null,
    estimate_cost: 2,
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

function makeBoard(
  id: string,
  cards: KanbanCard[],
  extraColumns: { id: string; name: string }[] = [],
): KanbanBoard {
  return {
    id,
    name: id,
    project_id: `/tmp/${id}`,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    columns: [
      {
        id: "col-backlog",
        board_id: id,
        name: "Backlog",
        position: 0,
        color: "#6b7280",
        cards,
      },
      {
        id: "col-progress",
        board_id: id,
        name: "In Progress",
        position: 1,
        color: "#3b82f6",
        cards: [],
      },
      ...extraColumns.map((column, index) => ({
        id: column.id,
        board_id: id,
        name: column.name,
        position: 2 + index,
        color: null,
        cards: [] as KanbanCard[],
      })),
    ],
  };
}

describe("kanban swimlanes", () => {
  it("returns no lanes when a board has no grouping", () => {
    const board = makeBoard("board-alpha", [makeCard()]);
    expect(buildBoardSwimlanes({ board, lanes: [] })).toEqual([]);
  });

  it("groups cards into lanes and an ungrouped remainder", () => {
    const auth = makeCard({ id: "auth", lane_id: "lane-auth" });
    const leftover = makeCard({ id: "other", title: "Other" });
    const board = makeBoard("board-alpha", [auth, leftover]);
    const lanes: KanbanLane[] = [
      {
        id: "lane-auth",
        boardId: "board-alpha",
        parentId: null,
        name: "Auth",
        kind: "remote",
        sourceId: "linear-1",
        remoteKey: "auth",
        position: 0,
      },
    ];

    const result = buildBoardSwimlanes({ board, lanes });
    expect(result.map((lane) => lane.id)).toEqual([
      "lane-auth",
      `board-alpha:${UNGROUPED_LANE_ID}`,
    ]);
    expect(result[0]?.cards.map((card) => card.id)).toEqual(["auth"]);
    expect(result[1]?.cards.map((card) => card.id)).toEqual(["other"]);
  });

  it("nests remote groups under workspace swimlanes in the combined view", () => {
    const alphaCard = makeCard({
      id: "alpha-auth",
      board_id: "board-alpha",
      lane_id: "lane-auth",
    });
    const betaCard = makeCard({
      id: "beta-task",
      board_id: "board-beta",
      column_id: "col-backlog",
      title: "Beta task",
    });
    const alpha = makeBoard("board-alpha", [alphaCard]);
    const beta = makeBoard("board-beta", [betaCard]);
    const lanes: KanbanLane[] = [
      {
        id: "lane-auth",
        boardId: "board-alpha",
        parentId: null,
        name: "Auth",
        kind: "remote",
        sourceId: "linear-1",
        remoteKey: "auth",
        position: 0,
      },
    ];

    const result = buildWorkspaceSwimlanes({
      boards: [
        { board: alpha, workspaceName: "alpha" },
        { board: beta, workspaceName: "beta" },
      ],
      lanes,
    });

    expect(result.map((lane) => lane.name)).toEqual(["alpha", "beta"]);
    expect(result[0]?.children.map((child) => child.name)).toEqual(["Auth"]);
    expect(laneCards(result[0]!).map((card) => card.id)).toEqual([
      "alpha-auth",
    ]);
    expect(laneCards(result[1]!).map((card) => card.id)).toEqual(["beta-task"]);
  });

  it("aligns columns across boards by normalized name", () => {
    const alpha = makeBoard("board-alpha", []);
    const beta = makeBoard(
      "board-beta",
      [],
      [{ id: "col-blocked", name: "Blocked" }],
    );

    expect(
      alignBoardColumns([alpha, beta]).map((column) => column.key),
    ).toEqual(["backlog", "in progress", "blocked"]);
  });
});
