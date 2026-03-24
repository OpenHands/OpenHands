import { useMemo } from "react";
import { OrganizationUserRole } from "#/types/org";
import { rolePermissions, PermissionKey } from "#/utils/org/permissions";

export const usePermission = (role: OrganizationUserRole) => {
  /* Memoize permissions for the role */
  const currentPermissions = useMemo<PermissionKey[]>(
    () => rolePermissions[role],
    [role],
  );

  /* Check if the user has a specific permission */
  const hasPermission = (permission: PermissionKey): boolean =>
    currentPermissions.includes(permission);

  return { hasPermission };
};
