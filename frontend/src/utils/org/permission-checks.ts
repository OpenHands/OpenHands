import { organizationService } from "#/api/organization-service/organization-service.api";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import {
  OrganizationMember,
  OrganizationsQueryData,
  OrganizationUserRole,
} from "#/types/org";
import { PermissionKey } from "./permissions";
import { queryClient } from "#/query-client-config";

/**
 * Get the active organization user.
 * Uses React Query's fetchQuery to leverage request deduplication,
 * preventing duplicate API calls when multiple consumers request the same data.
 *
 * On page refresh, the Zustand store resets and orgId becomes null.
 * In this case, we retrieve the organization from the query cache or fetch it
 * from the backend to ensure permission checks work correctly.
 *
 * @returns OrganizationMember
 */
export const getActiveOrganizationUser = async (): Promise<
  OrganizationMember | undefined
> => {
  let orgId = getSelectedOrganizationIdFromStore();

  // If no orgId in store (e.g., after page refresh), try to get it from query cache or fetch
  if (!orgId) {
    // Check if organizations data is already in the cache
    let organizationsData = queryClient.getQueryData<OrganizationsQueryData>([
      "organizations",
    ]);

    // If not in cache, fetch it
    if (!organizationsData) {
      try {
        organizationsData = await queryClient.fetchQuery({
          queryKey: ["organizations"],
          queryFn: organizationService.getOrganizations,
          staleTime: 1000 * 60 * 5, // 5 minutes - matches useOrganizations hook
        });
      } catch {
        return undefined;
      }
    }

    // Use currentOrgId from backend (user's last selected org) or first org as fallback
    orgId =
      organizationsData?.currentOrgId ??
      organizationsData?.items?.[0]?.id ??
      null;
  }

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
