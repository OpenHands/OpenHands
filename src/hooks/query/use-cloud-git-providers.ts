import { useQuery } from "@tanstack/react-query";
import { listCloudGitProviders } from "#/api/cloud/git-providers-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * The cloud instance's git provider rows and whether the user linked each.
 * Only an instance with `enable_oauth_v2_login` serves them, so callers
 * enable the query from that flag.
 */
export function useCloudGitProviders({ enabled }: { enabled: boolean }) {
  const { backend, orgId } = useActiveBackend();

  return useQuery({
    queryKey: [
      "cloud-git-providers",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: listCloudGitProviders,
    enabled: enabled && backend.kind === "cloud",
    staleTime: 1000 * 60,
    retry: false,
    meta: { disableToast: true },
  });
}
