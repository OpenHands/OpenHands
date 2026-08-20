import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { Branch } from "#/types/git";
import { Provider } from "#/types/settings";

export function useSearchBranches(
  repository: string | null,
  query: string,
  perPage: number = 30,
  selectedProvider?: Provider,
) {
  return useQuery<Branch[]>({
    queryKey: [
      "repository",
      repository,
      "branches",
      "search",
      query,
      perPage,
      selectedProvider,
    ],
    queryFn: async () => {
      if (!repository || !query || !selectedProvider) return [];
      const response = await GitService.searchRepositoryBranches(
        repository,
        selectedProvider,
        query,
        undefined, // pageId
        perPage,
      );
      return response.items;
    },
    enabled: !!repository && !!query && !!selectedProvider,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
}
