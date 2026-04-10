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

  const repos = useInfiniteQuery<RepositoryPage>({
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
        const { installationIndex, pageId } = pageParam as {
          installationIndex: number | null;
          pageId: string | null;
        };

        if (!installations) {
          throw new Error("Missing installation list");
        }

        const result = await GitService.retrieveInstallationRepositories(
          provider,
          installationIndex || 0,
          installations,
          pageId ?? undefined,
          pageSize,
        );
        return result;
      }

      // Use type assertion to ensure correct type
      const pageParamString = pageParam as string | null | undefined;
      const resolvedPageId = pageParamString ?? undefined;
      const result = await GitService.retrieveUserGitRepositories(
        provider,
        resolvedPageId,
        pageSize,
      );
      return result;
    },
    getNextPageParam: (lastPage) => lastPage.next_page_id,
    initialPageParam: useInstallationRepos
      ? { installationIndex: 0, pageId: null as string | null }
      : (null as string | null),
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
