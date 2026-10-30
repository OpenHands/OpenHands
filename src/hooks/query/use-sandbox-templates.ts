import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { getSandboxSelectionOptions } from "#/api/cloud/sandbox-template-service.api";
import { SANDBOX_TEMPLATES_QUERY_KEYS } from "./query-keys";

export function useSandboxTemplates() {
  const { backend, orgId } = useActiveBackend();
  return useQuery({
    // Resolve credentials at execution time; keep only scope IDs in the cache.
    queryKey: [...SANDBOX_TEMPLATES_QUERY_KEYS.all, backend.id, orgId] as const,
    queryFn: ({ queryKey }) =>
      getSandboxSelectionOptions(queryKey[1], queryKey[2]!),
    enabled: backend.kind === "cloud" && Boolean(orgId),
    staleTime: 0,
    refetchOnWindowFocus: true,
    retry: false,
    // Older backends have no selector endpoint; keep their launch UI unchanged.
    meta: { disableToast: true },
  });
}
