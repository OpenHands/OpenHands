import { describe, expect, it } from "vitest";
import { suggestedTaskToWorkItem } from "#/utils/suggested-task-work-item";
import type { SuggestedTask } from "#/utils/types";

const openIssue: SuggestedTask = {
  git_provider: "github",
  issue_number: 142,
  repo: "owner/repo",
  title: "Mission Provider Filtering",
  task_type: "OPEN_ISSUE",
};

describe("suggestedTaskToWorkItem", () => {
  it("maps an OPEN_ISSUE task to a provider-neutral reference", () => {
    expect(suggestedTaskToWorkItem(openIssue)).toEqual({
      provider: "github",
      id: "142",
      title: "Mission Provider Filtering",
      repository: "owner/repo",
    });
  });

  it.each(["MERGE_CONFLICTS", "FAILING_CHECKS", "UNRESOLVED_COMMENTS"] as const)(
    "does not name a conversation after a pull-request chore (%s)",
    (task_type) => {
      expect(suggestedTaskToWorkItem({ ...openIssue, task_type })).toBeNull();
    },
  );

  it("returns null when there is no task", () => {
    expect(suggestedTaskToWorkItem(null)).toBeNull();
    expect(suggestedTaskToWorkItem(undefined)).toBeNull();
  });

  it("carries the provider's own id verbatim as a string", () => {
    const gitlab = { ...openIssue, git_provider: "gitlab" as const };
    expect(suggestedTaskToWorkItem(gitlab)).toMatchObject({
      provider: "gitlab",
      id: "142",
    });
  });
});
