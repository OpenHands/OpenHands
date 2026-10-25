import { ConversationStatus } from "#/types/conversation-status";
import { RuntimeStatus } from "#/types/runtime-status";
import { Provider } from "#/types/settings";

export interface FileUploadSuccessResponse {
  uploaded_files: string[];
  skipped_files: { name: string; reason: string }[];
}

export interface GetVSCodeUrlResponse {
  vscode_url: string | null;
  error?: string;
}

export interface GetTrajectoryResponse {
  trajectory: unknown[] | null;
  error?: string;
}

export interface RepositorySelection {
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: Provider | null;
}

export type ConversationTrigger =
  | "resolver"
  | "gui"
  | "suggested_task"
  | "microagent_management"
  | "automation";

export interface Conversation {
  conversation_id: string;
  title: string;
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: Provider | null;
  last_updated_at: string;
  created_at: string;
  status: ConversationStatus;
  runtime_status: RuntimeStatus | null;
  trigger?: ConversationTrigger;
  url: string | null;
  session_api_key: string | null;
  pr_number?: number[] | null;
  conversation_version?: "V0" | "V1";
  sub_conversation_ids?: string[];
  public?: boolean;
  llm_model?: string | null;
}

/**
 * @deprecated Use AgentServerGitChangeStatus for new code. This type is maintained for backward compatibility with V0 API.
 */
export type GitChangeStatus = "M" | "A" | "D" | "R" | "U";

export type AgentServerGitChangeStatus =
  | "MOVED"
  | "ADDED"
  | "DELETED"
  | "UPDATED";

export interface GitChange {
  status: GitChangeStatus;
  path: string;
}

export interface GitChangeDiff {
  modified: string;
  original: string;
}

export interface GitCommit {
  sha: string;
  shortSha: string;
  subject: string;
  author: string;
  // ISO 8601 author date with UTC offset (git log %aI).
  timestamp: string;
}

export interface GitCommitsPage {
  commits: GitCommit[];
  hasMore: boolean;
}
