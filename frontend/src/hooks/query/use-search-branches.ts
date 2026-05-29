import { useInfiniteQuery, InfiniteData } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { BranchPage } from "#/types/git";
import { Provider } from "#/types/settings";

export function useSearchBranches(
  repository: string | null,
  query: string,
  perPage: number = 30,
  selectedProvider?: Provider,
) {
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
      "search",
      query,
      perPage,
      selectedProvider,
    ],
    queryFn: async ({ pageParam }) => {
      if (!repository || !query || !selectedProvider) {
        return {
          items: [],
          next_page_id: null,
        };
      }
      return GitService.getRepositoryBranches(
        repository,
        selectedProvider,
        query,
        pageParam ?? undefined,
        perPage,
      );
    },
    enabled: !!repository && !!query && !!selectedProvider,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    getNextPageParam: (lastPage) =>
      lastPage.next_page_id ? lastPage.next_page_id : undefined,
    initialPageParam: null,
  });

  return {
    ...result,
    data: result.data?.pages.flatMap((page) => page.items) ?? [],
  };
}
