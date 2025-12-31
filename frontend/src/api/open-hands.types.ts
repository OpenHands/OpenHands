import { ConversationStatus } from "#/types/conversation-status";
import { RuntimeStatus } from "#/types/runtime-status";
import { Provider } from "#/types/settings";

export interface ErrorResponse {
  error: string;
}

export interface SaveFileSuccessResponse {
  message: string;
}

export interface FileUploadSuccessResponse {
  uploaded_files: string[];
  skipped_files: { name: string; reason: string }[];
}

export interface FeedbackBodyResponse {
  message: string;
  feedback_id: string;
  password: string;
}

export interface FeedbackResponse {
  statusCode: number;
  body: FeedbackBodyResponse;
}

export interface AuthenticationResponse {
  message: string;
  login?: string; // Only present when allow list is enabled
}

export interface Feedback {
  version: string;
  email: string;
  token: string;
  polarity: "positive" | "negative";
  permissions: "public" | "private";
  trajectory: unknown[];
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
  | "microagent_management";

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
}

export interface ResultSet<T> {
  results: T[];
  next_page_id: string | null;
}

/**
 * @deprecated Use V1GitChangeStatus for new code. This type is maintained for backward compatibility with V0 API.
 */
export type GitChangeStatus = "M" | "A" | "D" | "R" | "U";

export type V1GitChangeStatus = "MOVED" | "ADDED" | "DELETED" | "UPDATED";

export interface GitChange {
  status: GitChangeStatus;
  path: string;
}

export interface GitChangeDiff {
  modified: string;
  original: string;
}

export interface InputMetadata {
  name: string;
  description: string;
}

export interface Microagent {
  name: string;
  type: "repo" | "knowledge";
  content: string;
  triggers: string[];
}

export interface GetMicroagentsResponse {
  microagents: Microagent[];
}

export interface GetMicroagentPromptResponse {
  status: string;
  prompt: string;
}

export interface IOption<T> {
  label: string;
  value: T;
}

export interface CreateMicroagent {
  repo: string;
  git_provider?: Provider;
  title?: string;
}

export interface MicroagentContentResponse {
  content: string;
  path: string;
  git_provider: Provider;
  triggers: string[];
}

export type GetFilesResponse = string[];

export interface GetFileResponse {
  code: string;
}
