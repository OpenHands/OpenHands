import { http, HttpResponse } from "msw";
import {
  DEFAULT_PROJECT_BOARD_NAME,
  KANBAN_API_BOARDS_PATH,
  KANBAN_API_CARDS_PATH,
  KANBAN_API_COLUMNS_PATH,
  PROJECT_API_INIT_PATH,
  PROJECT_API_PREVIEW_PATH,
} from "#/api/kanban-service/kanban-constants";
import type {
  KanbanBoard,
  KanbanBoardCosts,
  KanbanCard,
  KanbanColumn,
  SuggestedKanbanCard,
} from "#/api/kanban-service/kanban-types";

const DEFAULT_COLUMNS = [
  { name: "Backlog", color: "#6b7280" },
  { name: "In Progress", color: "#3b82f6" },
  { name: "Review", color: "#f59e0b" },
  { name: "Done", color: "#22c55e" },
];

let boards: KanbanBoard[] = [];
let nextId = 1;

function id(prefix: string): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

function now(): string {
  return new Date().toISOString();
}

function suggestedFromSpec(spec?: string): SuggestedKanbanCard {
  const title = spec?.trim().split("\n")[0] || "Detected work";
  return {
    title,
    description: spec?.trim() || title,
    source: spec?.trim() ? "decomposition" : "readme",
    acceptance: ["Spec is implemented"],
    priority: "P2",
  };
}

function createBoard(
  name: string,
  projectId: string | null = null,
): KanbanBoard {
  const createdAt = now();
  const boardId = id("board");
  return {
    id: boardId,
    name,
    project_id: projectId,
    created_at: createdAt,
    updated_at: createdAt,
    columns: DEFAULT_COLUMNS.map((column, index) => ({
      id: id("col"),
      board_id: boardId,
      name: column.name,
      position: index,
      color: column.color,
      cards: [],
    })),
  };
}

export function resetKanbanMockData() {
  boards = [];
  nextId = 1;
}

function getBoard(boardId: string): KanbanBoard | undefined {
  return boards.find((board) => board.id === boardId);
}

function boardCosts(board: KanbanBoard): KanbanBoardCosts {
  const columns = board.columns.map((column) => {
    const cards = column.cards ?? [];
    return {
      id: column.id,
      name: column.name,
      estimate_cost: cards.reduce(
        (sum, card) => sum + (card.estimate_cost ?? 0),
        0,
      ),
      actual_cost: cards.reduce(
        (sum, card) => sum + (card.actual_cost ?? 0),
        0,
      ),
      estimate_tokens: cards.reduce(
        (sum, card) => sum + (card.estimate_tokens ?? 0),
        0,
      ),
      actual_tokens: cards.reduce(
        (sum, card) => sum + (card.actual_tokens ?? 0),
        0,
      ),
    };
  });
  return {
    board_id: board.id,
    total_estimate_cost: columns.reduce(
      (sum, column) => sum + column.estimate_cost,
      0,
    ),
    total_actual_cost: columns.reduce(
      (sum, column) => sum + column.actual_cost,
      0,
    ),
    total_estimate_tokens: columns.reduce(
      (sum, column) => sum + column.estimate_tokens,
      0,
    ),
    total_actual_tokens: columns.reduce(
      (sum, column) => sum + column.actual_tokens,
      0,
    ),
    columns,
  };
}

export const KANBAN_HANDLERS = [
  http.post(`*${PROJECT_API_PREVIEW_PATH}`, async ({ request }) => {
    const body = (await request.json()) as { spec?: string };
    return HttpResponse.json({ suggested: [suggestedFromSpec(body.spec)] });
  }),
  http.post(`*${PROJECT_API_INIT_PATH}`, async ({ request }) => {
    const body = (await request.json()) as {
      spec?: string;
      board_name?: string;
    };
    const suggested = suggestedFromSpec(body.spec);
    const board = createBoard(body.board_name ?? DEFAULT_PROJECT_BOARD_NAME);
    const backlog = board.columns[0];
    const createdAt = now();
    const card: KanbanCard = {
      id: id("card"),
      column_id: backlog.id,
      board_id: board.id,
      title: suggested.title,
      description: suggested.description,
      priority: suggested.priority,
      status: "todo",
      assignee: null,
      linked_branch: null,
      linked_pr: null,
      estimate_tokens: null,
      estimate_cost: null,
      actual_tokens: null,
      actual_cost: null,
      model_used: null,
      tool_calls: null,
      agent_time: null,
      agent_session_id: null,
      position: 0,
      created_at: createdAt,
      updated_at: createdAt,
    };
    backlog.cards = [card];
    boards.push(board);
    return HttpResponse.json(
      { suggested: [suggested], board, cards: [card] },
      { status: 201 },
    );
  }),
  http.get(`*${KANBAN_API_BOARDS_PATH}`, () =>
    HttpResponse.json(
      boards.map(({ columns: _columns, ...summary }) => summary),
    ),
  ),
  http.post(`*${KANBAN_API_BOARDS_PATH}`, async ({ request }) => {
    const body = (await request.json()) as {
      name?: string;
      project_id?: string;
    };
    const board = createBoard(body.name ?? "Board", body.project_id ?? null);
    boards.push(board);
    return HttpResponse.json(board, { status: 201 });
  }),
  http.get(`*${KANBAN_API_BOARDS_PATH}/:boardId/costs`, ({ params }) => {
    const board = getBoard(String(params.boardId));
    if (!board)
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    return HttpResponse.json(boardCosts(board));
  }),
  http.get(`*${KANBAN_API_BOARDS_PATH}/:boardId`, ({ params }) => {
    const board = getBoard(String(params.boardId));
    if (!board)
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    return HttpResponse.json(board);
  }),
  http.post(
    `*${KANBAN_API_BOARDS_PATH}/:boardId/columns`,
    async ({ params, request }) => {
      const board = getBoard(String(params.boardId));
      if (!board)
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      const body = (await request.json()) as { name?: string; color?: string };
      const column: KanbanColumn = {
        id: id("col"),
        board_id: board.id,
        name: body.name ?? "Column",
        position: board.columns.length,
        color: body.color ?? null,
        cards: [],
      };
      board.columns.push(column);
      return HttpResponse.json(column, { status: 201 });
    },
  ),
  http.post(
    `*${KANBAN_API_COLUMNS_PATH}/:columnId/cards`,
    async ({ params, request }) => {
      const columnId = String(params.columnId);
      const board = boards.find((item) =>
        item.columns.some((column) => column.id === columnId),
      );
      const column = board?.columns.find((item) => item.id === columnId);
      if (!board || !column) {
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      }
      const body = (await request.json()) as Partial<KanbanCard> & {
        title?: string;
      };
      const createdAt = now();
      const card: KanbanCard = {
        id: id("card"),
        column_id: columnId,
        board_id: board.id,
        title: body.title ?? "Card",
        description: body.description ?? null,
        priority: body.priority ?? "P2",
        status: body.status ?? "todo",
        assignee: body.assignee ?? null,
        linked_branch: null,
        linked_pr: null,
        estimate_tokens: body.estimate_tokens ?? null,
        estimate_cost: body.estimate_cost ?? null,
        actual_tokens: null,
        actual_cost: null,
        model_used: null,
        tool_calls: null,
        agent_time: null,
        agent_session_id: null,
        position: (column.cards ?? []).length,
        created_at: createdAt,
        updated_at: createdAt,
      };
      column.cards = [...(column.cards ?? []), card];
      return HttpResponse.json(card, { status: 201 });
    },
  ),
  http.patch(
    `*${KANBAN_API_CARDS_PATH}/:cardId`,
    async ({ params, request }) => {
      const cardId = String(params.cardId);
      const body = (await request.json()) as Partial<KanbanCard>;
      for (const board of boards) {
        for (const column of board.columns) {
          const card = (column.cards ?? []).find((item) => item.id === cardId);
          if (card) {
            Object.assign(card, body, { updated_at: now() });
            return HttpResponse.json(card);
          }
        }
      }
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    },
  ),
  http.delete(`*${KANBAN_API_CARDS_PATH}/:cardId`, ({ params }) => {
    const cardId = String(params.cardId);
    for (const board of boards) {
      for (const column of board.columns) {
        const before = column.cards ?? [];
        column.cards = before.filter((card) => card.id !== cardId);
        if (column.cards.length !== before.length) {
          return new HttpResponse(null, { status: 204 });
        }
      }
    }
    return HttpResponse.json({ error: "not found" }, { status: 404 });
  }),
  http.post(
    `*${KANBAN_API_CARDS_PATH}/:cardId/move`,
    async ({ params, request }) => {
      const cardId = String(params.cardId);
      const body = (await request.json()) as {
        column_id: string;
        position: number;
      };
      let moved: KanbanCard | null = null;
      for (const board of boards) {
        for (const column of board.columns) {
          const index = (column.cards ?? []).findIndex(
            (card) => card.id === cardId,
          );
          if (index >= 0) {
            [moved] = (column.cards ?? []).splice(index, 1);
          }
        }
        if (!moved) continue;
        const dest = board.columns.find(
          (column) => column.id === body.column_id,
        );
        if (!dest)
          return HttpResponse.json({ error: "not found" }, { status: 404 });
        dest.cards = dest.cards ?? [];
        dest.cards.splice(body.position, 0, {
          ...moved,
          column_id: dest.id,
          position: body.position,
          updated_at: now(),
        });
        return HttpResponse.json(dest.cards[body.position]);
      }
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    },
  ),
];
