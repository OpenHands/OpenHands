import { useQuery } from "@tanstack/react-query";
import { isAgentServerVersionError } from "@openhands/typescript-client/clients";

import WorkspacesService, {
  WorkspacesListResponse,
} from "#/api/workspaces-service/workspaces-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";

interface UseLocalWorkspacesOptions {
  enabled?: boolean;
}

export function useLocalWorkspaces({
  enabled = true,
}: UseLocalWorkspacesOptions = {}) {
  const active = useActiveBackend();
  return useQuery<WorkspacesListResponse>({
    queryKey: LOCAL_WORKSPACES_QUERY_KEYS.byBackend(
      active.backend.id,
      active.orgId,
    ),
    queryFn: () => WorkspacesService.listWorkspaces(),
    enabled,
    retry: (failureCount, error) =>
      !isAgentServerVersionError(error) && failureCount < 3,
    meta: { disableToast: true },
    staleTime: 60_000,
  });
}
