import { expect, test } from "vitest";
import { SuggestedTask, SuggestedTaskGroup } from "#/utils/types";
import { groupSuggestedTasks } from "#/utils/group-suggested-tasks";

const rawTasks: SuggestedTask[] = [
  {
    issue_number: 1,
    repo: "repo1",
    title: "Task 1",
    task_type: "MERGE_CONFLICTS",
    git_provider: "github",
  },
  {
    issue_number: 2,
    repo: "repo1",
    title: "Task 2",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 3,
    repo: "repo2",
    title: "Task 3",
    task_type: "UNRESOLVED_COMMENTS",
    git_provider: "github",
  },
  {
    issue_number: 4,
    repo: "repo2",
    title: "Task 4",
    task_type: "OPEN_ISSUE",
    git_provider: "github",
  },
  {
    issue_number: 5,
    repo: "repo3",
    title: "Task 5",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
];

const groupedTasks: SuggestedTaskGroup[] = [
  {
    title: "repo1",
    tasks: [
      {
        issue_number: 1,
        repo: "repo1",
        title: "Task 1",
        task_type: "MERGE_CONFLICTS",
        git_provider: "github",
      },
      {
        issue_number: 2,
        repo: "repo1",
        title: "Task 2",
        task_type: "FAILING_CHECKS",
        git_provider: "github",
      },
    ],
  },
  {
    title: "repo2",
    tasks: [
      {
        issue_number: 3,
        repo: "repo2",
        title: "Task 3",
        task_type: "UNRESOLVED_COMMENTS",
        git_provider: "github",
      },
      {
        issue_number: 4,
        repo: "repo2",
        title: "Task 4",
        task_type: "OPEN_ISSUE",
        git_provider: "github",
      },
    ],
  },
  {
    title: "repo3",
    tasks: [
      {
        issue_number: 5,
        repo: "repo3",
        title: "Task 5",
        task_type: "FAILING_CHECKS",
        git_provider: "github",
      },
    ],
  },
];

test("groupSuggestedTasks", () => {
  expect(groupSuggestedTasks(rawTasks)).toEqual(groupedTasks);
});
