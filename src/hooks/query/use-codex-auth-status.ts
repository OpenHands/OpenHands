import { useQuery } from "@tanstack/react-query";
import { CodexAuthService } from "#/api/codex-auth-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CODEX_AUTH_QUERY_KEYS } from "./query-keys";

export function useCodexAuthStatus(enabled = true) {
  const { backend } = useActiveBackend();
  // eslint-disable-next-line @tanstack/query/exhaustive-deps -- Backend id/revision identify the connection without putting its session key in cache keys.
  return useQuery({
    queryKey: CODEX_AUTH_QUERY_KEYS.status(
      backend.id,
      backend.connectionRevision,
    ),
    queryFn: () => CodexAuthService.getStatus(backend),
    enabled: enabled && backend.kind === "local",
    staleTime: 0,
    refetchInterval: enabled ? 30000 : false,
    retry: false,
    meta: { disableToast: true, backendId: backend.id },
  });
}
