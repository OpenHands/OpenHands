import { Provider } from "#/types/settings";

export type SuggestedTaskType =
  | "MERGE_CONFLICTS"
  | "FAILING_CHECKS"
  | "UNRESOLVED_COMMENTS"
  | "OPEN_ISSUE"; // This is a task type identifier, not a UI string

export interface SuggestedTask {
  git_provider: Provider;
  issue_number: number;
  repo: string;
  title: string;
  task_type: SuggestedTaskType;
}

export interface SuggestedTaskGroup {
  title: string;
  tasks: SuggestedTask[];
}
