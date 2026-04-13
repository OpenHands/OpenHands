import { useInfiniteQuery, InfiniteData } from "@tanstack/react-query";
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

type InstallationCursor = { installationIndex: number; pageId: string | null };
type UserCursor = string | null;
type Cursor = InstallationCursor | UserCursor;

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
    InfiniteData<RepositoryPage>,
    [string, string[], Provider | null, boolean, number, ...unknown[]],
    Cursor
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

        const cursor = pageParam as InstallationCursor;
        const result = await GitService.retrieveInstallationRepositories(
          provider,
          cursor.installationIndex,
          installations,
          cursor.pageId ?? undefined,
          pageSize,
        );
        return result;
      }

      const cursor = pageParam as UserCursor;
      const result = await GitService.retrieveUserGitRepositories(
        provider,
        cursor ?? undefined,
        pageSize,
      );
      return result;
    },
    getNextPageParam: (lastPage, allPages, lastPageParam) => {
      if (useInstallationRepos && installations) {
        // Installation-based pagination
        const currentCursor = lastPageParam as InstallationCursor;
        if (lastPage.next_page_id) {
          return {
            installationIndex: currentCursor.installationIndex,
            pageId: lastPage.next_page_id,
          };
        }
        // Advance to next installation
        const nextInstallationIndex = currentCursor.installationIndex + 1;
        if (nextInstallationIndex < installations.length) {
          return { installationIndex: nextInstallationIndex, pageId: null };
        }
        return undefined;
      }
      // User repositories pagination
      return lastPage.next_page_id;
    },
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
