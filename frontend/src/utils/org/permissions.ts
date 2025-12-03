import { OrganizationUserRole } from "#/types/org";

type UserRoleChangePermissionKey = "change_user_role";
type InviteUserToOrganizationKey = "invite_user_to_organization";

type ChangeUserRolePermission =
  `${UserRoleChangePermissionKey}:${OrganizationUserRole}`;

type ChangeOrganizationNamePermission = "change_organization_name";
type DeleteOrganizationPermission = "delete_organization";
type AddCreditsPermission = "add_credits";

type UserPermission =
  | InviteUserToOrganizationKey
  | ChangeUserRolePermission
  | ChangeOrganizationNamePermission
  | DeleteOrganizationPermission
  | AddCreditsPermission;

const ownerPerms: UserPermission[] = [
  "invite_user_to_organization",
  "change_organization_name",
  "delete_organization",
  "add_credits",
  "change_user_role:owner",
  "change_user_role:admin",
  "change_user_role:user",
];
const adminPerms: UserPermission[] = [
  "invite_user_to_organization",
  "change_user_role:user",
];
const userPerms: UserPermission[] = [];

export const rolePermissions: Record<OrganizationUserRole, UserPermission[]> = {
  owner: ownerPerms,
  admin: adminPerms,
  user: userPerms,
};
