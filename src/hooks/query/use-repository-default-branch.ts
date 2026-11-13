import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import { Provider } from "#/types/settings";

/**
 * The default branch of a repository whose list entry does not carry one
 * (e.g. Bitbucket Data Center). A failed lookup leaves it unknown.
 */
export const useRepositoryDefaultBranch = (
  repository: string | null,
  provider: Provider,
  enabled: boolean,
) =>
  useQuery({
    queryKey: ["repository", repository, "default-branch", provider],
    queryFn: () => GitService.getRepositoryDefaultBranch(repository!, provider),
    enabled: enabled && !!repository && !!provider,
    staleTime: 1000 * 60 * 5, // 5 minutes
    retry: false,
    meta: { disableToast: true },
  });
