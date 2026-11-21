import { useEffect } from "react";
import {
  useQueries,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import type { ProjectPullRequest } from "#/api/projects-service/projects-service.api";
import {
  getPullRequestStatus,
  type PullRequestStatus,
} from "#/api/git-service/pull-request-status.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { isUserIdle, watchUserActivity } from "#/utils/user-activity";

/** `owner/repo#number`, the key a pull request goes by in Canvas. */
export const pullRequestRef = (pr: ProjectPullRequest) =>
  `${pr.owner}/${pr.repo}#${pr.number}`;

const PULL_REQUEST_STATUS = "pull-request-status";

/** How old a pull request's state may get before it is read again. */
export const PULL_REQUEST_STATUS_FRESH_MS = 5 * 60_000;

/** How often the shared refresher looks for states that went stale. */
const REFRESH_CHECK_MS = 60_000;

let refresher: { users: number; stop: () => void } | undefined;

/**
 * One refresher for every pull request state on screen, however many places
 * show it: it reads again only the states older than
 * {@link PULL_REQUEST_STATUS_FRESH_MS}, and only while the user is at the
 * tab. When they come back, stale states are read at once.
 */
function keepPullRequestStatusesFresh(queryClient: QueryClient): () => void {
  if (!refresher) {
    const refresh = () =>
      queryClient.refetchQueries(
        { queryKey: [PULL_REQUEST_STATUS], type: "active", stale: true },
        { cancelRefetch: false },
      );
    const timer = setInterval(() => {
      if (!isUserIdle()) refresh();
    }, REFRESH_CHECK_MS);
    const stopWatching = watchUserActivity(refresh);
    refresher = {
      users: 0,
      stop: () => {
        clearInterval(timer);
        stopWatching();
      },
    };
  }
  const current = refresher;
  current.users += 1;
  return () => {
    current.users -= 1;
    if (current.users === 0) {
      current.stop();
      if (refresher === current) refresher = undefined;
    }
  };
}

/**
 * The live state of each pull request, keyed by {@link pullRequestRef}. Each
 * state is one request shared by every caller, read again at most every
 * {@link PULL_REQUEST_STATUS_FRESH_MS} and not while the user is away. A pull
 * request whose state cannot be read (no provider token, rate limit) is
 * missing from the map.
 */
export function usePullRequestStatuses(
  pullRequests: ProjectPullRequest[],
): Map<string, PullRequestStatus> {
  const { backend } = useActiveBackend();
  const queryClient = useQueryClient();
  useEffect(() => keepPullRequestStatusesFresh(queryClient), [queryClient]);
  return useQueries({
    queries: pullRequests.map((pr) => ({
      queryKey: [PULL_REQUEST_STATUS, backend.id, pullRequestRef(pr)],
      queryFn: () => getPullRequestStatus(pr.owner, pr.repo, pr.number),
      staleTime: PULL_REQUEST_STATUS_FRESH_MS,
      refetchOnWindowFocus: false,
      retry: false,
      meta: { disableToast: true },
    })),
    combine: (results) => {
      const statuses = new Map<string, PullRequestStatus>();
      results.forEach((result, i) => {
        if (result.data)
          statuses.set(pullRequestRef(pullRequests[i]), result.data);
      });
      return statuses;
    },
  });
}
