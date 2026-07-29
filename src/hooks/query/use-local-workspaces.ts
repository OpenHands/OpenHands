import { useQuery } from "@tanstack/react-query";
import { isAgentServerVersionError } from "@openhands/typescript-client/clients";

import WorkspacesService, {
  WorkspacesListResponse,
} from "#/api/workspaces-service/workspaces-service.api";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";

interface UseLocalWorkspacesOptions {
  enabled?: boolean;
}

export function useLocalWorkspaces({
  enabled = true,
}: UseLocalWorkspacesOptions = {}) {
  return useQuery<WorkspacesListResponse>({
    queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
    queryFn: () => WorkspacesService.listWorkspaces(),
    enabled,
    retry: (failureCount, error) =>
      !isAgentServerVersionError(error) && failureCount < 3,
    meta: { disableToast: true },
    staleTime: 60_000,
  });
}
