import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { GitProviderPullRequestStatusService } from "#/api/git-provider-pull-request-status-service";
import { mockJsonResponse } from "./cloud/fetch-test-utils";

// A cloud backend keeps provider tokens server-side, so the provider REST
// calls are the only fetches in play.
const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const API = "https://api.github.com/repos/acme/widgets";

const originalFetch = global.fetch;
const fetchMock = vi.fn();

type Route = Response | (() => Response);

/** Answers each request by its exact URL, so parallel calls can't race. */
function routeFetch(routes: Record<string, Route>) {
  fetchMock.mockImplementation(async (url: string) => {
    const route = routes[url];
    if (!route) {
      throw new Error(`Unexpected fetch: ${url}`);
    }
    return typeof route === "function" ? route() : route.clone();
  });
}

function pull(
  number: number,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    number,
    title: `PR ${number}`,
    html_url: `https://github.com/acme/widgets/pull/${number}`,
    state: "open",
    draft: false,
    merged_at: null,
    mergeable: true,
    mergeable_state: "clean",
    updated_at: "2026-10-28T10:00:00Z",
    head: {
      sha: `sha${number}`,
      ref: `feature-${number}`,
      repo: { full_name: "acme/widgets", default_branch: "main" },
    },
    ...overrides,
  };
}

/** Routes for one PR: detail, check runs, commit statuses, reviews. */
function prRoutes(
  number: number,
  {
    detail = pull(number),
    checkRuns = [] as Record<string, unknown>[],
    checkRunsTotal,
    statuses = [] as Record<string, unknown>[],
    reviews = [] as Record<string, unknown>[],
  }: {
    detail?: Record<string, unknown>;
    checkRuns?: Record<string, unknown>[];
    checkRunsTotal?: number;
    statuses?: Record<string, unknown>[];
    reviews?: Record<string, unknown>[];
  } = {},
): Record<string, Route> {
  const sha = (detail.head as { sha: string }).sha;
  return {
    [`${API}/pulls/${number}`]: mockJsonResponse(detail),
    [`${API}/commits/${sha}/check-runs?per_page=100`]: mockJsonResponse({
      total_count: checkRunsTotal ?? checkRuns.length,
      check_runs: checkRuns,
    }),
    [`${API}/commits/${sha}/status?per_page=100`]: mockJsonResponse({
      total_count: statuses.length,
      statuses,
    }),
    [`${API}/pulls/${number}/reviews?per_page=100`]: mockJsonResponse(reviews),
  };
}

const run = (status: string, conclusion: string | null = null) => ({
  status,
  conclusion,
});
const review = (login: string, state: string) => ({
  user: { login },
  state,
});

function load(input: {
  branch?: string | null;
  prNumbers?: number[];
  provider?: "github" | "gitlab";
}) {
  return GitProviderPullRequestStatusService.getConversationPullRequests({
    repository: "acme/widgets",
    provider: input.provider ?? "github",
    branch: input.branch ?? null,
    prNumbers: input.prNumbers ?? [],
  });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend]);
  setActiveSelection({ backendId: cloudBackend.id });
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  global.fetch = originalFetch;
});

describe("GitProviderPullRequestStatusService", () => {
  it("reports lifecycle, checks, review and conflicts separately for a tracked PR", async () => {
    // Arrange: one failing check run beside a passing commit status, and an
    // approval, on a PR that has merge conflicts.
    routeFetch(
      prRoutes(7, {
        detail: pull(7, { mergeable: false, mergeable_state: "dirty" }),
        checkRuns: [run("completed", "success"), run("completed", "failure")],
        statuses: [{ state: "success" }],
        reviews: [review("octocat", "APPROVED")],
      }),
    );

    // Act
    const result = await load({ prNumbers: [7] });

    // Assert
    expect(result).toEqual({
      status: "ok",
      source: "tracked",
      omittedCount: 0,
      pullRequests: [
        {
          repository: "acme/widgets",
          number: 7,
          title: "PR 7",
          url: "https://github.com/acme/widgets/pull/7",
          lifecycle: "open",
          checks: "failing",
          review: "approved",
          mergeable: "conflicting",
          updatedAt: "2026-10-28T10:00:00Z",
          unavailableReason: null,
        },
      ],
    });
  });

  it("uses each reviewer's latest decision and reports running checks as pending", async () => {
    // Arrange: octocat approves after requesting changes, hubot still wants
    // changes, and comments do not count as a decision.
    routeFetch(
      prRoutes(8, {
        detail: pull(8, {
          draft: true,
          mergeable: null,
          mergeable_state: "unknown",
        }),
        checkRuns: [run("completed", "success"), run("in_progress")],
        reviews: [
          review("octocat", "CHANGES_REQUESTED"),
          review("hubot", "CHANGES_REQUESTED"),
          review("octocat", "APPROVED"),
          review("hubot", "COMMENTED"),
        ],
      }),
    );

    // Act
    const result = await load({ prNumbers: [8] });

    // Assert
    expect(result.status === "ok" && result.pullRequests[0]).toMatchObject({
      lifecycle: "draft",
      checks: "pending",
      review: "changes_requested",
      mergeable: "unknown",
    });
  });

  it("tells merged from closed-without-merging and shows no checks as none", async () => {
    // Arrange
    routeFetch({
      ...prRoutes(9, {
        detail: pull(9, { state: "closed", merged_at: "2026-10-27T09:00:00Z" }),
        reviews: [
          review("octocat", "APPROVED"),
          review("octocat", "DISMISSED"),
        ],
      }),
      ...prRoutes(10, { detail: pull(10, { state: "closed" }) }),
    });

    // Act
    const result = await load({ prNumbers: [9, 10] });

    // Assert
    expect(
      result.status === "ok" &&
        result.pullRequests.map(
          ({ number, lifecycle, checks, review: r, mergeable }) => ({
            number,
            lifecycle,
            checks,
            review: r,
            mergeable,
          }),
        ),
    ).toEqual([
      {
        number: 9,
        lifecycle: "merged",
        checks: "none",
        review: "none",
        mergeable: "not_applicable",
      },
      {
        number: 10,
        lifecycle: "closed",
        checks: "none",
        review: "none",
        mergeable: "not_applicable",
      },
    ]);
  });

  it("does not report passing checks when some check runs were not fetched", async () => {
    // Arrange: 100 successes shown out of 150; the unseen 50 could fail.
    routeFetch({
      ...prRoutes(11, {
        checkRuns: Array.from({ length: 100 }, () =>
          run("completed", "success"),
        ),
        checkRunsTotal: 150,
      }),
      ...prRoutes(12, {
        checkRuns: [run("completed", "timed_out")],
        checkRunsTotal: 150,
      }),
    });

    // Act
    const result = await load({ prNumbers: [11, 12] });

    // Assert: a failure already seen is certain; a pass is not.
    expect(
      result.status === "ok" && result.pullRequests.map((pr) => pr.checks),
    ).toEqual(["unknown", "failing"]);
  });

  it("keeps an unreadable PR in place as unknown instead of dropping it", async () => {
    // Arrange: #13 is gone or private, #14 hit the rate limit, #15 is fine.
    routeFetch({
      [`${API}/pulls/13`]: mockJsonResponse({ message: "Not Found" }, 404),
      [`${API}/pulls/14`]: () =>
        new Response("{}", {
          status: 403,
          headers: { "x-ratelimit-remaining": "0" },
        }),
      ...prRoutes(15),
    });

    // Act
    const result = await load({ prNumbers: [13, 14, 15] });

    // Assert
    expect(result.status).toBe("ok");
    const prs = result.status === "ok" ? result.pullRequests : [];
    expect(
      prs.map((pr) => [pr.number, pr.lifecycle, pr.unavailableReason]),
    ).toEqual([
      [13, "unknown", "not_found"],
      [14, "unknown", "rate_limited"],
      [15, "open", null],
    ]);
    expect(prs[0]).toMatchObject({
      url: "https://github.com/acme/widgets/pull/13",
      checks: "unknown",
      review: "unknown",
      mergeable: "unknown",
    });
  });

  it("caps tracked PRs and reports how many were left out", async () => {
    // Arrange: duplicates collapse, then only the first ten are fetched.
    const numbers = Array.from({ length: 12 }, (_, i) => 100 + i);
    routeFetch(Object.assign({}, ...numbers.map((n) => prRoutes(n))));

    // Act
    const result = await load({ prNumbers: [...numbers, 100] });

    // Assert
    expect(
      result.status === "ok" && result.pullRequests.map((pr) => pr.number),
    ).toEqual(numbers.slice(0, 10));
    expect(result.status === "ok" && result.omittedCount).toBe(2);
  });

  it("falls back to PRs opened from the conversation branch, ignoring the default branch", async () => {
    // Arrange
    routeFetch({
      [`${API}/pulls?state=all&head=acme%3Afix-login&per_page=10`]:
        mockJsonResponse([
          pull(21, {
            head: {
              sha: "sha21",
              ref: "fix-login",
              repo: { default_branch: "main" },
            },
          }),
        ]),
      ...prRoutes(21, {
        detail: pull(21, {
          head: {
            sha: "sha21",
            ref: "fix-login",
            repo: { default_branch: "main" },
          },
        }),
      }),
      [`${API}/pulls?state=all&head=acme%3Amain&per_page=10`]: mockJsonResponse(
        [
          pull(22, {
            head: {
              sha: "sha22",
              ref: "main",
              repo: { default_branch: "main" },
            },
          }),
        ],
      ),
    });

    // Act
    const fromFeature = await load({ branch: "fix-login" });
    const fromDefault = await load({ branch: "main" });

    // Assert
    expect(fromFeature).toMatchObject({
      status: "ok",
      source: "branch",
      pullRequests: [{ number: 21, lifecycle: "open" }],
    });
    expect(fromDefault).toEqual({
      status: "ok",
      source: "branch",
      omittedCount: 0,
      pullRequests: [],
    });
  });

  it("makes no request when there is nothing to associate", async () => {
    // Act
    const result = await load({});

    // Assert
    expect(result).toEqual({
      status: "ok",
      source: "none",
      omittedCount: 0,
      pullRequests: [],
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports providers without status support as unsupported", async () => {
    // Act
    const result = await load({ provider: "gitlab", prNumbers: [1] });

    // Assert
    expect(result).toEqual({ status: "unsupported_provider" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports the branch lookup failing as unavailable, not as no PRs", async () => {
    // Arrange
    routeFetch({
      [`${API}/pulls?state=all&head=acme%3Afix-login&per_page=10`]:
        mockJsonResponse({ message: "Bad credentials" }, 401),
    });

    // Act
    const result = await load({ branch: "fix-login" });

    // Assert
    expect(result).toEqual({ status: "unavailable", reason: "auth_failed" });
  });
});
