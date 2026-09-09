import type {
  KanbanBoard,
  KanbanCard,
  KanbanColumn,
} from "#/api/kanban-service/kanban-types";
import type { KanbanLane } from "#/utils/kanban-sync";

export const UNGROUPED_LANE_ID = "ungrouped";
export const WORKSPACE_LANE_PREFIX = "workspace:";

const DEFAULT_COLUMN_ORDER = ["backlog", "in progress", "review", "done"];

export type SwimlaneKind = "workspace" | "manual" | "remote" | "ungrouped";

export interface SwimlaneNode {
  id: string;
  name: string;
  kind: SwimlaneKind;
  boardId: string;
  sourceId: string | null;
  depth: 0 | 1;
  cards: KanbanCard[];
  children: SwimlaneNode[];
}

export interface AlignedColumn {
  key: string;
  name: string;
}

export function flattenBoardCards(board: KanbanBoard): KanbanCard[] {
  return board.columns.flatMap((column) => column.cards ?? []);
}

export function normalizeColumnName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, " ");
}

export function alignBoardColumns(boards: KanbanBoard[]): AlignedColumn[] {
  const seen = new Map<string, string>();
  for (const board of boards) {
    for (const column of board.columns) {
      const key = normalizeColumnName(column.name);
      if (!seen.has(key)) seen.set(key, column.name);
    }
  }
  return [...seen.entries()]
    .sort(([left], [right]) => {
      const leftRank = DEFAULT_COLUMN_ORDER.indexOf(left);
      const rightRank = DEFAULT_COLUMN_ORDER.indexOf(right);
      if (leftRank >= 0 && rightRank >= 0) return leftRank - rightRank;
      if (leftRank >= 0) return -1;
      if (rightRank >= 0) return 1;
      return left.localeCompare(right);
    })
    .map(([key, name]) => ({ key, name }));
}

export function laneCards(lane: SwimlaneNode): KanbanCard[] {
  if (lane.children.length === 0) return lane.cards;
  return [...lane.cards, ...lane.children.flatMap(laneCards)];
}

function ungroupedLane(board: KanbanBoard, cards: KanbanCard[]): SwimlaneNode {
  return {
    id: `${board.id}:${UNGROUPED_LANE_ID}`,
    name: UNGROUPED_LANE_ID,
    kind: "ungrouped",
    boardId: board.id,
    sourceId: null,
    depth: 0,
    cards,
    children: [],
  };
}

export function buildBoardSwimlanes(args: {
  board: KanbanBoard;
  lanes: KanbanLane[];
}): SwimlaneNode[] {
  const cards = flattenBoardCards(args.board);
  const boardLanes = args.lanes
    .filter((lane) => lane.boardId === args.board.id)
    .slice()
    .sort((left, right) => left.position - right.position);
  if (boardLanes.length === 0) return [];

  const assigned = new Set<string>();
  const childrenOf = (parentId: string | null) =>
    boardLanes.filter((lane) => lane.parentId === parentId);

  const nodeFromLane = (lane: KanbanLane, depth: 0 | 1): SwimlaneNode => {
    const ownCards = cards.filter((card) => card.lane_id === lane.id);
    ownCards.forEach((card) => assigned.add(card.id));
    return {
      id: lane.id,
      name: lane.name,
      kind: lane.kind,
      boardId: args.board.id,
      sourceId: lane.sourceId,
      depth,
      cards: ownCards,
      children: childrenOf(lane.id).map((child) => nodeFromLane(child, 1)),
    };
  };

  const result = childrenOf(null).map((lane) => nodeFromLane(lane, 0));
  const leftover = cards.filter((card) => !assigned.has(card.id));
  if (leftover.length > 0) {
    result.push(ungroupedLane(args.board, leftover));
  }
  return result;
}

export function buildWorkspaceSwimlanes(args: {
  boards: { board: KanbanBoard; workspaceName: string }[];
  lanes: KanbanLane[];
}): SwimlaneNode[] {
  return args.boards.map(({ board, workspaceName }) => {
    const inner = buildBoardSwimlanes({ board, lanes: args.lanes }).map(
      (lane) => ({ ...lane, depth: 1 as const }),
    );
    return {
      id: `${WORKSPACE_LANE_PREFIX}${board.id}`,
      name: workspaceName,
      kind: "workspace",
      boardId: board.id,
      sourceId: null,
      depth: 0,
      cards: inner.length === 0 ? flattenBoardCards(board) : [],
      children: inner,
    };
  });
}

export function columnsForLane(
  board: KanbanBoard,
  aligned: AlignedColumn[],
  cards: KanbanCard[],
): KanbanColumn[] {
  return aligned.map((column) => {
    const match = board.columns.find(
      (item) => normalizeColumnName(item.name) === column.key,
    );
    return {
      id: match?.id ?? `${board.id}:${column.key}`,
      board_id: board.id,
      name: match?.name ?? column.name,
      position: match?.position ?? 0,
      color: match?.color ?? null,
      cards: match ? cards.filter((card) => card.column_id === match.id) : [],
    };
  });
}

export function costsFromBoards(boards: KanbanBoard[]): {
  board_id: string;
  total_estimate_cost: number;
  total_actual_cost: number;
  total_estimate_tokens: number;
  total_actual_tokens: number;
  columns: {
    id: string;
    name: string;
    estimate_cost: number;
    actual_cost: number;
    estimate_tokens: number;
    actual_tokens: number;
  }[];
} {
  const cards = boards.flatMap(flattenBoardCards);
  return {
    board_id: boards.map((board) => board.id).join(",") || "combined",
    total_estimate_cost: cards.reduce(
      (sum, card) => sum + Number(card.estimate_cost ?? 0),
      0,
    ),
    total_actual_cost: cards.reduce(
      (sum, card) => sum + Number(card.actual_cost ?? 0),
      0,
    ),
    total_estimate_tokens: cards.reduce(
      (sum, card) => sum + Number(card.estimate_tokens ?? 0),
      0,
    ),
    total_actual_tokens: cards.reduce(
      (sum, card) => sum + Number(card.actual_tokens ?? 0),
      0,
    ),
    columns: [],
  };
}
