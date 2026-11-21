/**
 * Bridge from the repo's existing backlog-launch contract to the
 * provider-neutral {@link WorkItemReference}.
 *
 * `SuggestedTask` is how Canvas already receives an explicit backlog work
 * item at launch (the issue-to-PR style flows). Only `OPEN_ISSUE` denotes an
 * actual work item; the review-oriented types (`UNRESOLVED_COMMENTS`,
 * `FAILING_CHECKS`, `MERGE_CONFLICTS`) are pull-request chores, not backlog
 * items, so they must not be named after a ticket. The association is read
 * from the typed field, never inferred from prompt text or a connected repo.
 *
 * Kept separate from `work-item-reference.ts` so the generic naming module
 * stays free of repo-specific launch types.
 */
import type { SuggestedTask } from "#/utils/types";
import type { WorkItemReference } from "#/utils/work-item-reference";

export function suggestedTaskToWorkItem(
  task: SuggestedTask | null | undefined,
): WorkItemReference | null {
  if (task?.task_type !== "OPEN_ISSUE") return null;
  const title = task.title?.trim();
  return {
    provider: task.git_provider,
    // The provider's own issue number, verbatim (a Jira-style key would ride
    // here unchanged if a future task type carried one).
    id: String(task.issue_number),
    title: title ?? "",
    repository: task.repo ?? null,
  };
}
