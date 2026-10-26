import { useQuery } from "@tanstack/react-query";
import { listCloudServiceAccounts } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SERVICE_ACCOUNT_OWNERS_FEATURE } from "#/constants/automation-form";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { useDeploymentCapabilities } from "./use-manifest-capabilities";

/**
 * Whether the service-account owner UI applies: a cloud team workspace whose
 * automation service advertises `serviceAccountOwners`, for a caller with
 * `manage_automations`. Hidden on local, on personal workspaces (the org id is
 * the user id), without the capability, and for members.
 */
export function useServiceAccountOwnersEnabled(): boolean {
  const { backend, orgId } = useActiveBackend();
  const { canManage, userId } = useAutomationPermissions();
  const capabilities = useDeploymentCapabilities();

  if (backend.kind !== "cloud" || !orgId || !userId || orgId === userId) {
    return false;
  }
  if (!capabilities.data?.features?.includes(SERVICE_ACCOUNT_OWNERS_FEATURE)) {
    return false;
  }
  return canManage;
}

/**
 * The org's service accounts, for the owner picker and for naming a
 * service-owned automation's owner. Disabled unless
 * {@link useServiceAccountOwnersEnabled}; the enterprise list needs
 * `manage_service_accounts`, which owners and admins hold along with
 * `manage_automations`.
 */
export function useServiceAccounts() {
  const { backend, orgId } = useActiveBackend();
  const enabled = useServiceAccountOwnersEnabled();

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  return useQuery({
    queryKey: [
      "cloud-service-accounts",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => listCloudServiceAccounts(orgId!, backend),
    enabled,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
}
