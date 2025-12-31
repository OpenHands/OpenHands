import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { Branch, PaginatedBranchesResponse } from "#/types/git";
import { Provider } from "#/types/settings";

export const useRepositoryBranches = (
  repository: string | null,
  selectedProvider?: Provider,
) =>
  useQuery<Branch[]>({
    queryKey: ["repository", repository, "branches", selectedProvider],
    queryFn: async () => {
      if (!repository) return [];
      const response = await GitService.getRepositoryBranches(
        repository,
        1,
        30,
        selectedProvider,
      );
      // Ensure we return an array even if the response is malformed
      return Array.isArray(response.branches) ? response.branches : [];
    },
    enabled: !!repository,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });

export const useRepositoryBranchesPaginated = (
  repository: string | null,
  perPage: number = 30,
  selectedProvider?: Provider,
) =>
  useInfiniteQuery<PaginatedBranchesResponse, Error>({
    queryKey: [
      "repository",
      repository,
      "branches",
      "paginated",
      perPage,
      selectedProvider,
    ],
    queryFn: async ({ pageParam = 1 }) => {
      if (!repository) {
        return {
          branches: [],
          has_next_page: false,
          current_page: 1,
          per_page: perPage,
          total_count: 0,
        };
      }
      return GitService.getRepositoryBranches(
        repository,
        pageParam as number,
        perPage,
        selectedProvider,
      );
    },
    enabled: !!repository,
    staleTime: 1000 * 60 * 5, // 5 minutes
    getNextPageParam: (lastPage) =>
      // Use the has_next_page flag from the API response
      lastPage.has_next_page ? lastPage.current_page + 1 : undefined,
    initialPageParam: 1,
  });
