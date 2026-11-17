import { useQueries } from "@tanstack/react-query";
import type { ProjectPullRequest } from "#/api/projects-service/projects-service.api";
import {
  getPullRequestStatus,
  type PullRequestStatus,
} from "#/api/git-service/pull-request-status.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/** `owner/repo#number`, the key a pull request goes by in Canvas. */
export const pullRequestRef = (pr: ProjectPullRequest) =>
  `${pr.owner}/${pr.repo}#${pr.number}`;

/**
 * The live state of each pull request, keyed by {@link pullRequestRef}. One
 * request per pull request, cached for a minute and shared by every caller;
 * a pull request whose state cannot be read (no provider token, rate limit)
 * is missing from the map.
 */
export function usePullRequestStatuses(
  pullRequests: ProjectPullRequest[],
): Map<string, PullRequestStatus> {
  const { backend } = useActiveBackend();
  return useQueries({
    queries: pullRequests.map((pr) => ({
      queryKey: ["pull-request-status", backend.id, pullRequestRef(pr)],
      queryFn: () => getPullRequestStatus(pr.owner, pr.repo, pr.number),
      staleTime: 60_000,
      refetchInterval: 60_000,
      refetchIntervalInBackground: false,
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
