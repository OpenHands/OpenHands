import { useQuery } from "@tanstack/react-query";
import { GitProviderItemsService } from "#/api/git-provider-items-service";
import type { Provider } from "#/types/settings";

export function useRepositoryPullRequests(
  repository: string | null | undefined,
  provider: Provider | null | undefined,
) {
  return useQuery({
    queryKey: ["repository-pull-requests", provider, repository],
    queryFn: () =>
      GitProviderItemsService.listPullRequests(repository!, provider!),
    enabled: Boolean(repository && provider),
    staleTime: 60_000,
    meta: { disableToast: true },
  });
}

export function useRepositoryIssues(
  repository: string | null | undefined,
  provider: Provider | null | undefined,
) {
  return useQuery({
    queryKey: ["repository-issues", provider, repository],
    queryFn: () => GitProviderItemsService.listIssues(repository!, provider!),
    enabled: Boolean(repository && provider),
    staleTime: 60_000,
    meta: { disableToast: true },
  });
}
