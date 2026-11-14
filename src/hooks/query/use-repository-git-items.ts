import { useQuery } from "@tanstack/react-query";
import { GitProviderItemsService } from "#/api/git-provider-items-service";
import { useSettings } from "#/hooks/query/use-settings";
import type { Provider } from "#/types/settings";

/**
 * Instance the repository lives on: the cloud connection's custom host, then
 * the workspace remote's host, then the self-hosted instance configured on a
 * local backend (e.g. `GITLAB_HOST`). `null` means the provider's public host.
 * `isResolved` turns true once the configured-host lookup (if any) settled,
 * so API lists do not hit the public host first.
 */
export function useGitProviderHost(
  provider: Provider | null | undefined,
  remoteHost: string | null | undefined,
) {
  const { data: settings } = useSettings();
  const knownHost =
    (provider ? settings?.provider_tokens_set?.[provider] : null) ||
    remoteHost ||
    null;
  const configuredHostQuery = useQuery({
    queryKey: ["git-provider-configured-host", provider],
    queryFn: () => GitProviderItemsService.resolveConfiguredBaseUrl(provider!),
    enabled: Boolean(provider) && !knownHost,
    staleTime: 60_000,
    meta: { disableToast: true },
  });

  return {
    host: knownHost ?? configuredHostQuery.data ?? null,
    isResolved:
      Boolean(knownHost) || !provider || !configuredHostQuery.isPending,
  };
}

export function useRepositoryPullRequests(
  repository: string | null | undefined,
  provider: Provider | null | undefined,
  host: string | null = null,
  isHostResolved = true,
) {
  return useQuery({
    queryKey: ["repository-pull-requests", provider, repository, host],
    queryFn: () =>
      GitProviderItemsService.listPullRequests(repository!, provider!, host),
    enabled: Boolean(repository && provider) && isHostResolved,
    staleTime: 60_000,
    meta: { disableToast: true },
  });
}

export function useRepositoryIssues(
  repository: string | null | undefined,
  provider: Provider | null | undefined,
  host: string | null = null,
  isHostResolved = true,
) {
  return useQuery({
    queryKey: ["repository-issues", provider, repository, host],
    queryFn: () =>
      GitProviderItemsService.listIssues(repository!, provider!, host),
    enabled: Boolean(repository && provider) && isHostResolved,
    staleTime: 60_000,
    meta: { disableToast: true },
  });
}
