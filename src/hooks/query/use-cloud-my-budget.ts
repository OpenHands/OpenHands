import { useQuery } from "@tanstack/react-query";
import { getCloudMyBudget } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * The caller's own budget in the active org, via
 * `GET /api/organizations/{orgId}/budgets/me`. Disabled on local backends and
 * until an org is active. A failure stays quiet: it means the backend has no
 * budget details to offer.
 */
export function useCloudMyBudget({ enabled }: { enabled: boolean }) {
  const { backend, orgId } = useActiveBackend();

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  return useQuery({
    queryKey: [
      "cloud-my-budget",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudMyBudget(orgId!, backend),
    enabled: enabled && backend.kind === "cloud" && !!orgId,
    retry: false,
    meta: { disableToast: true },
  });
}
