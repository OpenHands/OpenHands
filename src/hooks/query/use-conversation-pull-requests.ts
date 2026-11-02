import { useQuery } from "@tanstack/react-query";
import { GitProviderPullRequestStatusService } from "#/api/git-provider-pull-request-status-service";
import { CONVERSATION_PULL_REQUESTS_QUERY_KEYS } from "#/hooks/query/query-keys";
import type { Provider } from "#/types/settings";

interface UseConversationPullRequestsInput {
  repository: string | null | undefined;
  provider: Provider | null | undefined;
  branch: string | null | undefined;
  prNumbers: readonly number[] | null | undefined;
}

/**
 * Provider-side status of the PRs that belong to a conversation. It is
 * re-read on the usual staleness cadence, so a PR merged or a check that
 * fails after the conversation ended still shows up.
 */
export function useConversationPullRequests({
  repository,
  provider,
  branch,
  prNumbers,
}: UseConversationPullRequestsInput) {
  const numbers = prNumbers ?? [];
  return useQuery({
    queryKey: CONVERSATION_PULL_REQUESTS_QUERY_KEYS.detail(
      provider,
      repository,
      branch ?? null,
      numbers,
    ),
    queryFn: () =>
      GitProviderPullRequestStatusService.getConversationPullRequests({
        repository: repository!,
        provider: provider!,
        branch: branch ?? null,
        prNumbers: numbers,
      }),
    enabled: Boolean(repository && provider),
    staleTime: 60_000,
    meta: { disableToast: true },
  });
}
