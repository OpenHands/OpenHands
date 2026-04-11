import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { Branch, BranchPage } from "#/types/git";
import { Provider } from "#/types/settings";

export const useRepositoryBranches = (
  repository: string | null,
  selectedProvider?: Provider,
) =>
  useQuery<Branch[]>({
    queryKey: ["repository", repository, "branches", selectedProvider],
    queryFn: async () => {
      if (!repository || !selectedProvider) return [];
      const response = await GitService.getRepositoryBranches(
        repository,
        selectedProvider, // provider (required)
        "", // query (empty = list all)
        undefined, // pageId
        30, // limit
      );
      // Ensure we return an array even if the response is malformed
      return Array.isArray(response.items) ? response.items : [];
    },
    enabled: !!repository && !!selectedProvider,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });

export const useRepositoryBranchesPaginated = (
  repository: string | null,
  perPage: number = 30,
  selectedProvider?: Provider,
) =>
  useInfiniteQuery<BranchPage, Error>({
    queryKey: [
      "repository",
      repository,
      "branches",
      "paginated",
      perPage,
      selectedProvider,
    ],
    queryFn: async ({ pageParam = null as string | null }) => {
      if (!repository || !selectedProvider) {
        return {
          items: [],
          next_page_id: null,
        };
      }
      // Use type assertion to ensure correct type
      const pageParamString = pageParam as string | null | undefined;
      const resolvedPageId = pageParamString ?? undefined;
      return GitService.getRepositoryBranches(
        repository,
        selectedProvider,
        "", // query (empty = list all)
        resolvedPageId,
        perPage,
      );
    },
    enabled: !!repository && !!selectedProvider,
    staleTime: 1000 * 60 * 5, // 5 minutes
    getNextPageParam: (lastPage) =>
      // Use next_page_id from the cursor-based API
      lastPage.next_page_id ? lastPage.next_page_id : undefined,
    initialPageParam: null,
  });
