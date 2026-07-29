import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { GitRepository } from "#/types/git";
import { Provider } from "#/types/settings";

export function useSearchRepositories(
  query: string,
  selectedProvider?: Provider | null,
  disabled?: boolean,
  pageSize: number = 100,
) {
  // For backward compatibility, return the items array directly
  return useQuery<GitRepository[]>({
    queryKey: ["repositories", "search", query, selectedProvider, pageSize],
    queryFn: async () => {
      if (!selectedProvider) {
        return [];
      }
      const response = await GitService.searchGitRepositories(
        query,
        selectedProvider, // provider (required)
        pageSize,
      );
      return response.items;
    },
    enabled: !!query && !!selectedProvider && !disabled,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
}
