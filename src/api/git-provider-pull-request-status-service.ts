import {
  fetchGithubJson,
  GitProviderRequestError,
  resolveProviderToken,
} from "#/api/git-provider-items-service";
import type { Provider } from "#/types/settings";
import { constructPullRequestUrl } from "#/utils/utils";

export type PullRequestLifecycle =
  | "draft"
  | "open"
  | "merged"
  | "closed"
  | "unknown";

export type PullRequestChecksState =
  | "passing"
  | "pending"
  | "failing"
  | "none"
  | "unknown";

export type PullRequestReviewState =
  | "approved"
  | "changes_requested"
  | "none"
  | "unknown";

export type PullRequestMergeability =
  | "clean"
  | "conflicting"
  | "not_applicable"
  | "unknown";

export type GitProviderAccessProblem =
  | "not_found"
  | "auth_failed"
  | "forbidden"
  | "rate_limited"
  | "error";

/**
 * Where the conversation → PR association came from. `tracked` PRs are the
 * conversation's own `pr_number` list; `branch` PRs were opened from the
 * conversation's branch. PRs are never inferred from conversation text.
 */
export type PullRequestAssociationSource = "tracked" | "branch" | "none";

export interface ConversationPullRequest {
  repository: string;
  number: number;
  title: string | null;
  url: string;
  lifecycle: PullRequestLifecycle;
  checks: PullRequestChecksState;
  review: PullRequestReviewState;
  mergeable: PullRequestMergeability;
  updatedAt: string | null;
  /** Set when the PR itself could not be read; every state is then `unknown`. */
  unavailableReason: GitProviderAccessProblem | null;
}

export type ConversationPullRequestsResult =
  | {
      status: "ok";
      source: PullRequestAssociationSource;
      pullRequests: ConversationPullRequest[];
      /** Associated PRs beyond the per-conversation cap that were not fetched. */
      omittedCount: number;
    }
  | { status: "unsupported_provider" }
  | { status: "unavailable"; reason: GitProviderAccessProblem };

export interface ConversationPullRequestsInput {
  repository: string;
  provider: Provider;
  branch: string | null;
  prNumbers: readonly number[];
}

// Each PR costs four requests, and unauthenticated GitHub allows 60 an hour.
const MAX_PULL_REQUESTS = 10;
// GitHub's page-size ceiling; a fuller result than this was cut off.
const PAGE_SIZE = 100;

type GithubPull = {
  number: number;
  title: string;
  html_url: string;
  state: "open" | "closed";
  draft?: boolean;
  merged_at: string | null;
  mergeable?: boolean | null;
  mergeable_state?: string;
  updated_at?: string | null;
  head: {
    sha: string;
    ref: string;
    repo?: { default_branch?: string } | null;
  };
};

type GithubCheckRuns = {
  total_count: number;
  check_runs: { status: string; conclusion: string | null }[];
};

type GithubCombinedStatus = {
  total_count: number;
  statuses: { state: string }[];
};

type GithubReview = {
  user?: { login?: string } | null;
  state: string;
};

type CheckState = Exclude<PullRequestChecksState, "none">;

const PASSING_CONCLUSIONS = new Set(["success", "neutral", "skipped"]);
const FAILING_CONCLUSIONS = new Set([
  "failure",
  "timed_out",
  "cancelled",
  "action_required",
  "startup_failure",
]);

function checkRunState(
  checkRun: GithubCheckRuns["check_runs"][number],
): CheckState {
  if (checkRun.status !== "completed") return "pending";
  if (checkRun.conclusion && FAILING_CONCLUSIONS.has(checkRun.conclusion)) {
    return "failing";
  }
  if (checkRun.conclusion && PASSING_CONCLUSIONS.has(checkRun.conclusion)) {
    return "passing";
  }
  // e.g. `stale`: finished without a verdict we can stand behind.
  return "unknown";
}

function commitStatusState(state: string): CheckState {
  if (state === "success") return "passing";
  if (state === "failure" || state === "error") return "failing";
  if (state === "pending") return "pending";
  return "unknown";
}

/**
 * One failure decides the result, since it is certain. Otherwise any
 * unreadable or unfetched check could still fail, so the result is unknown
 * rather than a pass that was never observed.
 */
function aggregateChecks(
  states: CheckState[],
  truncated: boolean,
): PullRequestChecksState {
  if (states.includes("failing")) return "failing";
  if (truncated || states.includes("unknown")) return "unknown";
  if (states.includes("pending")) return "pending";
  return states.length > 0 ? "passing" : "none";
}

/** Reviews come oldest first; each reviewer's latest decision stands. */
function reviewState(reviews: GithubReview[]): PullRequestReviewState {
  if (reviews.length >= PAGE_SIZE) return "unknown";
  const decisions = new Map<string, string>();
  for (const { user, state } of reviews) {
    const login = user?.login;
    if (!login) continue;
    if (state === "DISMISSED") decisions.delete(login);
    else if (state === "APPROVED" || state === "CHANGES_REQUESTED") {
      decisions.set(login, state);
    }
  }
  const latest = [...decisions.values()];
  if (latest.includes("CHANGES_REQUESTED")) return "changes_requested";
  return latest.includes("APPROVED") ? "approved" : "none";
}

function lifecycleOf(pull: GithubPull): PullRequestLifecycle {
  if (pull.merged_at) return "merged";
  if (pull.state === "closed") return "closed";
  return pull.draft ? "draft" : "open";
}

function mergeabilityOf(pull: GithubPull): PullRequestMergeability {
  if (pull.state !== "open") return "not_applicable";
  if (pull.mergeable === false || pull.mergeable_state === "dirty") {
    return "conflicting";
  }
  // GitHub computes mergeability lazily and answers `null` until it has.
  return pull.mergeable === true ? "clean" : "unknown";
}

function accessProblemOf(error: unknown): GitProviderAccessProblem {
  if (!(error instanceof GitProviderRequestError)) return "error";
  if (error.rateLimited) return "rate_limited";
  if (error.status === 401) return "auth_failed";
  if (error.status === 403) return "forbidden";
  // GitHub answers 404 for private repositories the token cannot see.
  if (error.status === 404) return "not_found";
  return "error";
}

async function loadGithubPullRequest(
  repository: string,
  number: number,
  token: string | null,
): Promise<ConversationPullRequest> {
  const base = `/repos/${repository}`;
  try {
    const pull = await fetchGithubJson<GithubPull>(
      `${base}/pulls/${number}`,
      token,
    );
    const { sha } = pull.head;
    const [checkRuns, statuses, reviews] = await Promise.allSettled([
      fetchGithubJson<GithubCheckRuns>(
        `${base}/commits/${sha}/check-runs?per_page=${PAGE_SIZE}`,
        token,
      ),
      fetchGithubJson<GithubCombinedStatus>(
        `${base}/commits/${sha}/status?per_page=${PAGE_SIZE}`,
        token,
      ),
      fetchGithubJson<GithubReview[]>(
        `${base}/pulls/${number}/reviews?per_page=${PAGE_SIZE}`,
        token,
      ),
    ]);

    let checks: PullRequestChecksState = "unknown";
    if (checkRuns.status === "fulfilled" && statuses.status === "fulfilled") {
      const runs = checkRuns.value.check_runs;
      const commitStatuses = statuses.value.statuses;
      checks = aggregateChecks(
        [
          ...runs.map(checkRunState),
          ...commitStatuses.map((s) => commitStatusState(s.state)),
        ],
        checkRuns.value.total_count > runs.length ||
          statuses.value.total_count > commitStatuses.length,
      );
    }

    return {
      repository,
      number,
      title: pull.title,
      url:
        pull.html_url || constructPullRequestUrl(number, "github", repository),
      lifecycle: lifecycleOf(pull),
      checks,
      review:
        reviews.status === "fulfilled" ? reviewState(reviews.value) : "unknown",
      mergeable: mergeabilityOf(pull),
      updatedAt: pull.updated_at ?? null,
      unavailableReason: null,
    };
  } catch (error) {
    return {
      repository,
      number,
      title: null,
      url: constructPullRequestUrl(number, "github", repository),
      lifecycle: "unknown",
      checks: "unknown",
      review: "unknown",
      mergeable: "unknown",
      updatedAt: null,
      unavailableReason: accessProblemOf(error),
    };
  }
}

/**
 * PR numbers opened from `branch` in the same repository. A PR whose head is
 * the default branch is skipped: working on `main` is not evidence that the
 * conversation produced that PR.
 */
async function listGithubBranchPullRequests(
  repository: string,
  branch: string,
  token: string | null,
): Promise<number[]> {
  const [owner] = repository.split("/");
  const head = encodeURIComponent(`${owner}:${branch}`);
  const pulls = await fetchGithubJson<GithubPull[]>(
    `/repos/${repository}/pulls?state=all&head=${head}&per_page=${MAX_PULL_REQUESTS}`,
    token,
  );
  return pulls
    .filter((pull) => pull.head.ref !== pull.head.repo?.default_branch)
    .map((pull) => pull.number);
}

/**
 * Current lifecycle, checks, review and conflict state of the PRs that
 * belong to a conversation, read from the git provider so it stays accurate
 * after the conversation ends. Only GitHub is supported so far.
 */
export class GitProviderPullRequestStatusService {
  static async getConversationPullRequests({
    repository,
    provider,
    branch,
    prNumbers,
  }: ConversationPullRequestsInput): Promise<ConversationPullRequestsResult> {
    if (provider !== "github") {
      return { status: "unsupported_provider" };
    }
    const [owner, repo] = repository.split("/");
    if (!owner || !repo || (prNumbers.length === 0 && !branch)) {
      return {
        status: "ok",
        source: "none",
        pullRequests: [],
        omittedCount: 0,
      };
    }

    const token = await resolveProviderToken(provider);
    const source: PullRequestAssociationSource =
      prNumbers.length > 0 ? "tracked" : "branch";
    let numbers = [...new Set(prNumbers)];
    if (source === "branch") {
      try {
        numbers = await listGithubBranchPullRequests(
          repository,
          branch!,
          token,
        );
      } catch (error) {
        return { status: "unavailable", reason: accessProblemOf(error) };
      }
    }

    const fetched = numbers.slice(0, MAX_PULL_REQUESTS);
    const pullRequests = await Promise.all(
      fetched.map((number) => loadGithubPullRequest(repository, number, token)),
    );
    return {
      status: "ok",
      source,
      pullRequests,
      omittedCount: numbers.length - fetched.length,
    };
  }
}
