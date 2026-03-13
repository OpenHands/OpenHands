import { OrganizationUserRole } from "#/types/org";

/* PERMISSION TYPES */
type UserRoleChangePermissionKey = `change_user_role:${OrganizationUserRole}`;
type InviteUserToOrganizationKey = "invite_user_to_organization";

type ChangeOrganizationNamePermission = "change_organization_name";
type DeleteOrganizationPermission = "delete_organization";
type AddCreditsPermission = "add_credits";
type ViewBillingPermission = "view_billing";

type ManageSecretsPermission = "manage_secrets";
type ManageMCPPermission = "manage_mcp";
type ManageIntegrationsPermission = "manage_integrations";
type ManageApplicationSettingsPermission = "manage_application_settings";
type ManageAPIKeysPermission = "manage_api_keys";

type ViewLLMSettingsPermission = "view_llm_settings";
type EditLLMSettingsPermission = "edit_llm_settings";

// Union of all permission keys
export type PermissionKey =
  | UserRoleChangePermissionKey
  | InviteUserToOrganizationKey
  | ChangeOrganizationNamePermission
  | DeleteOrganizationPermission
  | AddCreditsPermission
  | ViewBillingPermission
  | ManageSecretsPermission
  | ManageMCPPermission
  | ManageIntegrationsPermission
  | ManageApplicationSettingsPermission
  | ManageAPIKeysPermission
  | ViewLLMSettingsPermission
  | EditLLMSettingsPermission;

/* PERMISSION ARRAYS */
const memberPerms: PermissionKey[] = [
  "manage_secrets",
  "manage_mcp",
  "manage_integrations",
  "manage_application_settings",
  "manage_api_keys",
  "view_llm_settings",
];

const adminOnly: PermissionKey[] = [
  "edit_llm_settings",
  "view_billing",
  "add_credits",
  "invite_user_to_organization",
  "change_user_role:member",
  "change_user_role:admin",
];

const ownerOnly: PermissionKey[] = [
  "change_organization_name",
  "delete_organization",
  "change_user_role:owner",
];

const adminPerms: PermissionKey[] = [...memberPerms, ...adminOnly];
const ownerPerms: PermissionKey[] = [...adminPerms, ...ownerOnly];

export const rolePermissions: Record<OrganizationUserRole, PermissionKey[]> = {
  owner: ownerPerms,
  admin: adminPerms,
  member: memberPerms,
};
