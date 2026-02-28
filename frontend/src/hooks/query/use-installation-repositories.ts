import { useInfiniteQuery } from "@tanstack/react-query";
import { useAppInstallations } from "./use-app-installations";
import { useConfig } from "./use-config";
import { useUserProviders } from "../use-user-providers";
import { Provider } from "#/types/settings";
import GitService from "#/api/git-service/git-service.api";
import { shouldUseInstallationRepos } from "#/utils/utils";

export const useInstallationRepositories = (
  selectedProvider: Provider | null,
) => {
  const { providers } = useUserProviders();
  const { data: config } = useConfig();
  const { data: installations } = useAppInstallations(selectedProvider);

  const repos = useInfiniteQuery({
    queryKey: [
      "repositories",
      providers || [],
      selectedProvider,
      installations || [],
    ],
    queryFn: async ({
      pageParam,
    }: {
      pageParam: { installationIndex: number | null; repoPage: number | null };
    }) => {
      const { repoPage, installationIndex } = pageParam;

      if (!installations) {
        throw new Error("Missing installation list");
      }

      return GitService.retrieveInstallationRepositories(
        selectedProvider!,
        installationIndex || 0,
        installations,
        repoPage || 1,
        30,
      );
    },
    initialPageParam: { installationIndex: 0, repoPage: 1 },
    getNextPageParam: (lastPage) => {
      if (lastPage.nextPage) {
        return {
          installationIndex: lastPage.installationIndex,
          repoPage: lastPage.nextPage,
        };
      }

      if (lastPage.installationIndex !== null) {
        return { installationIndex: lastPage.installationIndex, repoPage: 1 };
      }

      return null;
    },
    enabled:
      (providers || []).length > 0 &&
      !!selectedProvider &&
      shouldUseInstallationRepos(selectedProvider, config?.app_mode) &&
      Array.isArray(installations) &&
      installations.length > 0,
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
