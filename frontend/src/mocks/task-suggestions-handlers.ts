import { http, HttpResponse } from "msw";
import { SuggestedTask } from "#/utils/types";

const TASKS_1: SuggestedTask[] = [
  {
    issue_number: 6968,
    title: "Fix merge conflicts",
    repo: "octocat/hello-world",
    task_type: "MERGE_CONFLICTS",
    git_provider: "github",
  },
];

const TASKS_2: SuggestedTask[] = [
  {
    issue_number: 268,
    title: "Fix broken CI checks",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 281,
    title: "Fix issue",
    repo: "octocat/earth",
    task_type: "UNRESOLVED_COMMENTS",
    git_provider: "github",
  },
  {
    issue_number: 293,
    title: "Update documentation",
    repo: "octocat/earth",
    task_type: "OPEN_ISSUE",
    git_provider: "github",
  },
  {
    issue_number: 305,
    title: "Refactor user service",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 312,
    title: "Fix styling bug",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 327,
    title: "Add unit tests",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 331,
    title: "Implement dark mode",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 345,
    title: "Optimize build process",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
  {
    issue_number: 352,
    title: "Update dependencies",
    repo: "octocat/earth",
    task_type: "FAILING_CHECKS",
    git_provider: "github",
  },
];

export const MOCK_TASKS = [...TASKS_1, ...TASKS_2];

export const TASK_SUGGESTIONS_HANDLERS = [
  // New V1 endpoint with pagination
  http.get("/api/v1/git/suggested-tasks/search", async ({ request }) => {
    const url = new URL(request.url);
    const limit = url.searchParams.get("limit");
    const pageId = url.searchParams.get("page_id");

    // Simple pagination: return all items if no pagination params, otherwise apply limit
    let tasks = [...MOCK_TASKS];
    if (pageId || limit) {
      const limitNum = limit ? parseInt(limit, 10) : 30;
      tasks = tasks.slice(0, limitNum);
    }

    return HttpResponse.json({
      items: tasks,
      next_page_id: null, // No pagination in mock data
    });
  }),
  // Deprecated V0 endpoint (keep for backward compatibility)
  http.get("/api/user/suggested-tasks", async () =>
    HttpResponse.json(MOCK_TASKS),
  ),
];
