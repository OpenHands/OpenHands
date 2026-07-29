import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Server-defined permission (mirrors the app-server's
 * `authorization.Permission.EDIT_ORG_SETTINGS`) that gates org-scoped profile
 * mutations (both LLM profiles and agent profiles).
 */
const EDIT_ORG_SETTINGS = "edit_org_settings";

/**
 * Whether the current user may MUTATE org-scoped profiles — LLM profiles and
 * agent profiles alike — on the active backend: create, edit, rename, delete,
 * duplicate, or activate/switch.
 *
 * - Local agent-server (OSS): always `true`; the user owns their own profiles.
 * - Cloud: profiles are org-scoped. The app-server grants every mutating
 *   profile action (save/delete/rename/activate) only to the `owner`/`admin`
 *   roles via the `edit_org_settings` permission; a `member` is view-only. We
 *   read the caller's server-defined permissions from
 *   `GET /api/organizations/{orgId}/me` — the same call `useCloudCurrentUserId`
 *   makes, reusing its query key so no extra request is issued — and fall back
 *   to a role check when an older app-server doesn't return `permissions`.
 *
 * Returns `false` while the role/permissions are still loading or unknown on
 * cloud, so mutating controls never flash for a member before they resolve.
 */
export function useCanManageOrgProfiles(): boolean {
  const { backend, orgId } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  // `backend` is identified by `backend.id`, which is already in the key; we
  // keep the key byte-identical to useCloudCurrentUserId so React Query shares
  // the cached /me result instead of firing a second request.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const { data } = useQuery({
    queryKey: ["cloud-current-user", backend.id, orgId],
    queryFn: () => getCloudOrganizationMe(orgId!, backend),
    enabled: isCloud && !!orgId,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });

  if (!isCloud) return true;
  // Prefer the server-defined permission set; fall back to a role check for
  // older app-servers whose /me doesn't return `permissions` yet, so this
  // keeps working against either backend version.
  if (data?.permissions) {
    return data.permissions.includes(EDIT_ORG_SETTINGS);
  }
  return data?.role === "owner" || data?.role === "admin";
}
