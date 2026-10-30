import { useQuery } from "@tanstack/react-query";
import { listCloudServiceAccounts } from "#/api/cloud/organization-service.api";
import { fetchCloudIntegrationsConfig } from "#/api/cloud/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SERVICE_ACCOUNT_RUN_AS_FEATURE } from "#/constants/automation-form";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { useDeploymentCapabilities } from "./use-manifest-capabilities";

/**
 * Whether the service-account run-as UI applies: a cloud team workspace whose
 * automation service advertises `serviceAccountRunAs`, for a caller with
 * `manage_automations`. Hidden on local, on personal workspaces (the org id is
 * the user id), without the capability, and for members.
 */
export function useServiceAccountRunAsEnabled(): boolean {
  const { backend, orgId } = useActiveBackend();
  const { canManage, userId } = useAutomationPermissions();
  const isTeamWorkspace =
    backend.kind === "cloud" && !!orgId && !!userId && orgId !== userId;
  // Skip the capabilities request where the answer would be discarded.
  const capabilities = useDeploymentCapabilities({ enabled: isTeamWorkspace });
  // Same query as `useNativeGitIntegrations`, so both share one cached request.
  const { data: config } = useQuery({
    queryKey: [
      "cloud-providers-configured",
      backend.id,
      backend.connectionRevision ?? 0,
    ],
    queryFn: fetchCloudIntegrationsConfig,
    enabled: isTeamWorkspace,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

  if (!isTeamWorkspace) return false;
  if (config?.isServiceAccountsEnabled !== true) return false;
  if (!capabilities.data?.features?.includes(SERVICE_ACCOUNT_RUN_AS_FEATURE)) {
    return false;
  }
  return canManage;
}

/**
 * The org's service accounts, for the run-as picker and for naming the
 * service account an automation runs as. Disabled unless
 * {@link useServiceAccountRunAsEnabled}; the enterprise list needs
 * `manage_service_accounts`, which owners and admins hold along with
 * `manage_automations`.
 */
export function useServiceAccounts() {
  const { backend, orgId } = useActiveBackend();
  const enabled = useServiceAccountRunAsEnabled();

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
