import { useInfiniteQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import { useUserProviders } from "../use-user-providers";
import { Provider } from "#/types/settings";
import GitService from "#/api/git-service/git-service.api";
import { shouldUseInstallationRepos } from "#/utils/utils";

export const useUserRepositories = (selectedProvider: Provider | null) => {
  const { providers } = useUserProviders();
  const { data: config } = useConfig();

  const repos = useInfiniteQuery({
    queryKey: ["repositories", providers || [], selectedProvider],
    queryFn: async ({ pageParam }) =>
      GitService.retrieveUserGitRepositories(selectedProvider!, pageParam, 30),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => lastPage.nextPage,
    enabled:
      (providers || []).length > 0 &&
      !!selectedProvider &&
      !shouldUseInstallationRepos(selectedProvider, config?.app_mode),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });

  const onLoadMore = () => {
    if (repos.hasNextPage && !repos.isFetchingNextPage) {
      repos.fetchNextPage();
    }
  };

  // Return the query result with the scroll ref
  return {
    data: repos.data,
    isLoading: repos.isLoading,
    isError: repos.isError,
    hasNextPage: repos.hasNextPage,
    isFetchingNextPage: repos.isFetchingNextPage,
    onLoadMore,
  };
};
