import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Server-defined permission (mirrors the app-server's
 * `authorization.Permission.MANAGE_SUPER_ADMINS`) that an instance Super Admin
 * holds. The app-server lets only them add or remove an organization's Apps.
 */
const MANAGE_SUPER_ADMINS = "manage_super_admins";

/**
 * Whether the current user may add or remove Apps on the active backend.
 *
 * - Local agent-server (OSS): always `true`; the user owns their own Apps.
 * - Cloud: Apps are approved per organization by an instance Super Admin, and
 *   every other member only switches approved Apps on or off for themselves.
 *   The caller's permissions come from `GET /api/organizations/{orgId}/me`,
 *   the same call `useCloudCurrentUserId` makes, reusing its query key so no
 *   extra request is issued.
 *
 * Returns `false` while the permissions are still loading or unknown on cloud,
 * so the controls never flash for a member before they resolve.
 */
export function useCanManageCanvasApps(): boolean {
  const { backend, orgId } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  // Keep the key byte-identical to useCloudCurrentUserId so React Query shares
  // the cached /me result, including its connection-credential revision.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const { data } = useQuery({
    queryKey: [
      "cloud-current-user",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudOrganizationMe(orgId!, backend),
    enabled: isCloud && !!orgId,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

  if (!isCloud) return true;
  return data?.permissions?.includes(MANAGE_SUPER_ADMINS) === true;
}
