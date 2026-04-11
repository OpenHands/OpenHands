import { useInfiniteQuery, InfiniteData } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { BranchPage } from "#/types/git";
import { Provider } from "#/types/settings";

export const useRepositoryBranchesPaginated = (
  repository: string | null,
  perPage: number = 30,
  selectedProvider?: Provider,
) => {
  const result = useInfiniteQuery<
    BranchPage,
    Error,
    InfiniteData<BranchPage>,
    [string, string | null, ...unknown[]],
    string | null
  >({
    queryKey: [
      "repository",
      repository,
      "branches",
      "paginated",
      perPage,
      selectedProvider,
    ],
    queryFn: async ({ pageParam }) => {
      if (!repository || !selectedProvider) {
        return {
          items: [],
          next_page_id: null,
        };
      }
      return GitService.getRepositoryBranches(
        repository,
        selectedProvider,
        "", // query (empty = list all)
        pageParam ?? undefined,
        perPage,
      );
    },
    enabled: !!repository && !!selectedProvider,
    staleTime: 1000 * 60 * 5, // 5 minutes
    getNextPageParam: (lastPage) =>
      lastPage.next_page_id ? lastPage.next_page_id : undefined,
    initialPageParam: null,
  });

  return {
    ...result,
  };
};
