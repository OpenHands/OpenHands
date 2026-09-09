import { PROJECT_STATUSES, WORKTREE_STATUSES } from "./projects-constants";

export type ProjectStatus = (typeof PROJECT_STATUSES)[number];
export type WorktreeStatus = (typeof WORKTREE_STATUSES)[number];

export interface ProjectWorktree {
  id: string;
  project_id: string;
  branch_name: string;
  path: string;
  status: WorktreeStatus;
  agent_session_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  description: string | null;
  repo_url: string | null;
  local_path: string;
  default_branch: string;
  default_agent_profile: string | null;
  kanban_board_id: string | null;
  cost_cap: number | null;
  status: ProjectStatus;
  worktree_count: number;
  created_at: string;
  updated_at: string;
}

export interface Project extends ProjectSummary {
  worktrees: ProjectWorktree[];
}

export interface CreateProjectPayload {
  name: string;
  description?: string | null;
  repo_url?: string | null;
  local_path?: string | null;
  default_branch?: string | null;
  default_agent_profile?: string | null;
  kanban_board_id?: string | null;
  cost_cap?: number | null;
}

export type UpdateProjectPayload = Partial<
  Pick<
    ProjectSummary,
    | "name"
    | "description"
    | "default_branch"
    | "default_agent_profile"
    | "kanban_board_id"
    | "cost_cap"
    | "status"
  >
>;

export interface CreateWorktreePayload {
  branch_name: string;
  status?: WorktreeStatus;
}
