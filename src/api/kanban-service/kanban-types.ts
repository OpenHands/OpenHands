import { KANBAN_PRIORITIES } from "./kanban-constants";

export type KanbanPriority = (typeof KANBAN_PRIORITIES)[number];

export interface KanbanBoardSummary {
  id: string;
  name: string;
  project_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface KanbanCard {
  id: string;
  column_id: string;
  board_id: string;
  title: string;
  description: string | null;
  priority: KanbanPriority;
  status: string;
  assignee: string | null;
  linked_branch: string | null;
  linked_pr: string | null;
  estimate_tokens: number | null;
  estimate_cost: number | null;
  actual_tokens: number | null;
  actual_cost: number | null;
  model_used: string | null;
  tool_calls: number | null;
  agent_time: number | null;
  agent_session_id: string | null;
  position: number;
  created_at: string;
  updated_at: string;
}

export interface KanbanColumn {
  id: string;
  board_id: string;
  name: string;
  position: number;
  color: string | null;
  cards?: KanbanCard[];
}

export interface KanbanBoard extends KanbanBoardSummary {
  columns: KanbanColumn[];
}

export interface KanbanColumnCost {
  id: string;
  name: string;
  estimate_cost: number;
  actual_cost: number;
  estimate_tokens: number;
  actual_tokens: number;
}

export interface KanbanBoardCosts {
  board_id: string;
  total_estimate_cost: number;
  total_actual_cost: number;
  total_estimate_tokens: number;
  total_actual_tokens: number;
  columns: KanbanColumnCost[];
}

export interface CreateBoardPayload {
  name: string;
  project_id?: string | null;
}

export interface CreateColumnPayload {
  name: string;
  color?: string | null;
  position?: number | null;
}

export interface CreateCardPayload {
  title: string;
  description?: string | null;
  priority?: KanbanPriority;
  status?: string;
  assignee?: string | null;
  estimate_tokens?: number | null;
  estimate_cost?: number | null;
}

export interface MoveCardPayload {
  column_id: string;
  position: number;
}

export type UpdateCardPayload = Partial<
  Pick<
    KanbanCard,
    | "title"
    | "description"
    | "priority"
    | "status"
    | "assignee"
    | "linked_branch"
    | "linked_pr"
    | "estimate_tokens"
    | "estimate_cost"
    | "actual_tokens"
    | "actual_cost"
    | "model_used"
    | "tool_calls"
    | "agent_time"
    | "agent_session_id"
  >
>;
