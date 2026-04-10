import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { RepositoryPage } from "#/types/git";
import { Provider } from "#/types/settings";

export function useSearchRepositories(
  query: string,
  selectedProvider?: Provider | null,
  disabled?: boolean,
  pageSize: number = 100,
) {
  return useQuery<RepositoryPage>({
    queryKey: ["repositories", "search", query, selectedProvider, pageSize],
    queryFn: async () => {
      if (!selectedProvider) {
        return { items: [], next_page_id: null };
      }
      return GitService.searchGitRepositories(
        query,
        pageSize,
        selectedProvider, // provider (required)
      );
    },
    enabled: !!query && !!selectedProvider && !disabled,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
}
