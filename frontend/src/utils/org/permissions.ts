import { OrganizationUserRole } from "#/types/org";

type UserRoleChangePermissionKey = "change_user_role";
type InviteUserToOrganizationKey = "invite_user_to_organization";

type ChangeUserRolePermission =
  `${UserRoleChangePermissionKey}:${OrganizationUserRole}`;

type ChangeOrganizationNamePermission = "change_organization_name";
type DeleteOrganizationPermission = "delete_organization";

type UserPermission =
  | InviteUserToOrganizationKey
  | ChangeUserRolePermission
  | ChangeOrganizationNamePermission
  | DeleteOrganizationPermission;

const superadminPerms: UserPermission[] = [
  "invite_user_to_organization",
  "change_organization_name",
  "delete_organization",
  "change_user_role:superadmin",
  "change_user_role:admin",
  "change_user_role:user",
];
const adminPerms: UserPermission[] = [
  "invite_user_to_organization",
  "change_user_role:user",
];
const userPerms: UserPermission[] = [];

export const rolePermissions: Record<OrganizationUserRole, UserPermission[]> = {
  superadmin: superadminPerms,
  admin: adminPerms,
  user: userPerms,
};
