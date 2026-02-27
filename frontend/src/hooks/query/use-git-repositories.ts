import { useInfiniteQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import { useUserProviders } from "../use-user-providers";
import { useAppInstallations } from "./use-app-installations";
import { GitRepository } from "../../types/git";
import { Provider } from "../../types/settings";
import GitService from "#/api/git-service/git-service.api";
import { shouldUseInstallationRepos } from "#/utils/utils";

interface UseGitRepositoriesOptions {
  provider: Provider | null;
  pageSize?: number;
  enabled?: boolean;
}

interface UserRepositoriesResponse {
  data: GitRepository[];
  nextPage: number | null;
}

interface InstallationRepositoriesResponse {
  data: GitRepository[];
  nextPage: number | null;
  installationIndex: number | null;
}

export function useGitRepositories(options: UseGitRepositoriesOptions) {
  const { provider, pageSize = 30, enabled = true } = options;
  const { providers } = useUserProviders();
  const { data: config } = useConfig();
  const { data: installations } = useAppInstallations(provider);

  const useInstallationRepos = provider
    ? shouldUseInstallationRepos(provider, config?.app_mode)
    : false;

  const repos = useInfiniteQuery<
    UserRepositoriesResponse | InstallationRepositoriesResponse
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
        const { repoPage, installationIndex } = pageParam as {
          installationIndex: number | null;
          repoPage: number | null;
        };

        if (!installations) {
          throw new Error("Missing installation list");
        }

        return GitService.retrieveInstallationRepositories(
          provider,
          installationIndex || 0,
          installations,
          repoPage || 1,
          pageSize,
        );
      }

      return GitService.retrieveUserGitRepositories(
        provider,
        pageParam as number,
        pageSize,
      );
    },
    getNextPageParam: (lastPage) => {
      if (useInstallationRepos) {
        const installationPage = lastPage as InstallationRepositoriesResponse;
        if (installationPage.nextPage) {
          return {
            installationIndex: installationPage.installationIndex,
            repoPage: installationPage.nextPage,
          };
        }

        if (installationPage.installationIndex !== null) {
          return {
            installationIndex: installationPage.installationIndex,
            repoPage: 1,
          };
        }

        return null;
      }

      const userPage = lastPage as UserRepositoriesResponse;
      return userPage.nextPage;
    },
    initialPageParam: useInstallationRepos
      ? { installationIndex: 0, repoPage: 1 }
      : 1,
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
