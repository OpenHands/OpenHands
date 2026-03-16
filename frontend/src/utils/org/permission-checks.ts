import { organizationService } from "#/api/organization-service/organization-service.api";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { OrganizationMember, OrganizationUserRole } from "#/types/org";
import { PermissionKey } from "./permissions";
import { queryClient } from "#/query-client-config";

/**
 * Get the active organization user.
 * Uses React Query's fetchQuery to leverage request deduplication,
 * preventing duplicate API calls when multiple consumers request the same data.
 * @returns OrganizationMember
 */
export const getActiveOrganizationUser = async (): Promise<
  OrganizationMember | undefined
> => {
  const orgId = getSelectedOrganizationIdFromStore();
  if (!orgId) return undefined;

  try {
    const user = await queryClient.fetchQuery({
      queryKey: ["organizations", orgId, "me"],
      queryFn: () => organizationService.getMe({ orgId }),
      staleTime: 1000 * 60 * 5, // 5 minutes - matches useMe hook
    });
    return user;
  } catch {
    return undefined;
  }
};

/**
 * Get a list of roles that a user has permission to assign to other users
 * @param userPermissions all permission for active user
 * @returns an array of roles (strings) the user can change other users to
 */
export const getAvailableRolesAUserCanAssign = (
  userPermissions: PermissionKey[],
): OrganizationUserRole[] => {
  const availableRoles: OrganizationUserRole[] = [];
  if (userPermissions.includes("change_user_role:member")) {
    availableRoles.push("member");
  }
  if (userPermissions.includes("change_user_role:admin")) {
    availableRoles.push("admin");
  }
  if (userPermissions.includes("change_user_role:owner")) {
    availableRoles.push("owner");
  }
  return availableRoles;
};
