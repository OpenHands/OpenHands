import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudCurrentUserId } from "./query/use-cloud-current-user-id";
import type { Automation } from "#/types/automation";

/**
 * Server-defined permission strings (mirror the automation service's
 * `view_automations` / `manage_automations` and the enterprise
 * `authorization.Permission` enum). The `/me` endpoint returns whichever
 * subset the caller's role grants.
 */
export const VIEW_AUTOMATIONS = "view_automations";
export const MANAGE_AUTOMATIONS = "manage_automations";

export interface AutomationPermissionsResult {
  /** Read-only access (list, get, list runs, capabilities, git-sync status). */
  canView: boolean;
  /** Full write access (create, update, delete, dispatch, git-sync config). */
  canManage: boolean;
  /**
   * `true` when permissions are still being resolved on cloud. Local always
   * resolves synchronously. Mutating controls should stay hidden while this is
   * `true` so a member never sees a flash of write controls before the
   * server-defined permission set arrives.
   */
  isLoading: boolean;
}

/**
 * Resolve the caller's automation-level permissions on the active backend.
 *
 * - Local agent-server (OSS): always `{ canView: true, canManage: true }`.
 * - Cloud: reads the server-defined `permissions` array from
 *   `GET /api/organizations/{orgId}/me` (same query `useCloudCurrentUserId`
 *   and `useCanManageOrgProfiles` use, so no extra request is issued) and
 *   falls back to a role check (`owner`/`admin` ⇒ both, `member` ⇒ view
 *   only) when an older app-server doesn't return `permissions`.
 */
export function useAutomationPermissions(): AutomationPermissionsResult {
  const { backend, orgId } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const { data, isLoading } = useQuery({
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

  if (!isCloud) {
    return { canView: true, canManage: true, isLoading: false };
  }

  if (isLoading || !data) {
    return { canView: false, canManage: false, isLoading };
  }

  if (data.permissions) {
    const canView = data.permissions.includes(VIEW_AUTOMATIONS);
    const canManage = data.permissions.includes(MANAGE_AUTOMATIONS);
    return { canView, canManage, isLoading: false };
  }

  // Fall back to role check for older app-servers without `permissions`.
  const isOwnerOrAdmin = data.role === "owner" || data.role === "admin";
  return {
    canView: isOwnerOrAdmin || data.role === "member",
    canManage: isOwnerOrAdmin,
    isLoading: false,
  };
}

/**
 * Resolve whether the current user is the creator of the given automation.
 *
 * - Local: always `true` (the local user owns every automation; and since
 *   local always has `manage_automations`, the distinction is moot).
 * - Cloud: compares `automation.user_id` with the caller's `user_id` from
 *   `GET /api/organizations/{orgId}/me`. Returns `false` while the id is
 *   still loading so the creator escape hatch never flashes open prematurely.
 */
export function useIsAutomationOwner(automation: Automation): boolean {
  const { backend } = useActiveBackend();
  // Always call the hook so rules-of-hooks is satisfied; it's a no-op for
  // local backends since the early return below short-circuits before use.
  const userIds = useCloudCurrentUserId();

  if (backend.kind !== "cloud") return true;

  const entry = userIds[backend.id];
  if (!entry || entry.isLoading || !entry.userId) return false;
  return automation.user_id === entry.userId;
}
