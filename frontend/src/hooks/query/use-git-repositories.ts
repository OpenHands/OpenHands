import { useInfiniteQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import { useUserProviders } from "../use-user-providers";
import { useAppInstallations } from "./use-app-installations";
import { RepositoryPage } from "../../types/git";
import { Provider } from "../../types/settings";
import GitService from "#/api/git-service/git-service.api";
import { shouldUseInstallationRepos } from "#/utils/utils";

interface UseGitRepositoriesOptions {
  provider: Provider | null;
  pageSize?: number;
  enabled?: boolean;
}

export function useGitRepositories(options: UseGitRepositoriesOptions) {
  const { provider, pageSize = 30, enabled = true } = options;
  const { providers } = useUserProviders();
  const { data: config } = useConfig();
  const { data: page } = useAppInstallations(provider);
  const installations = page?.items;

  const useInstallationRepos = provider
    ? shouldUseInstallationRepos(provider, config?.app_mode)
    : false;

  const repos = useInfiniteQuery<
    RepositoryPage,
    Error,
    RepositoryPage,
    [string, string[], Provider | null, boolean, number, ...unknown[]],
    string | { installationIndex: number; pageId: string | null }
  >({
    queryKey: [
      "repositories",
      providers || [],
      provider,
      useInstallationRepos,
      pageSize,
      ...(useInstallationRepos ? [installations || []] : []),
    ],
    queryFn: async ({ pageParam }) => {
      if (!provider) {
        throw new Error("Provider is required");
      }

      if (useInstallationRepos) {
        if (!installations) {
          throw new Error("Missing installation list");
        }

        const { installationIndex, pageId } = pageParam as {
          installationIndex: number;
          pageId: string | null;
        };
        const result = await GitService.retrieveInstallationRepositories(
          provider,
          installationIndex || 0,
          installations,
          pageId ?? undefined,
          pageSize,
        );
        return result;
      }

      const pageId = pageParam as string | null;
      const result = await GitService.retrieveUserGitRepositories(
        provider,
        pageId ?? undefined,
        pageSize,
      );
      return result;
    },
    getNextPageParam: (lastPage) => lastPage.next_page_id,
    initialPageParam: useInstallationRepos
      ? { installationIndex: 0, pageId: null }
      : null,
    enabled:
      enabled &&
      (providers || []).length > 0 &&
      !!provider &&
      (!useInstallationRepos ||
        (Array.isArray(installations) && installations.length > 0)),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    refetchOnWindowFocus: false,
  });

  const onLoadMore = () => {
    if (repos.hasNextPage && !repos.isFetchingNextPage) {
      repos.fetchNextPage();
    }
  };

  return {
    data: repos.data,
    isLoading: repos.isLoading,
    isError: repos.isError,
    hasNextPage: repos.hasNextPage,
    isFetchingNextPage: repos.isFetchingNextPage,
    fetchNextPage: repos.fetchNextPage,
    onLoadMore,
  };
}
