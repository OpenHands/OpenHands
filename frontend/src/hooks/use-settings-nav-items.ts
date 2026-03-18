import { useConfig } from "#/hooks/query/use-config";
import {
  SAAS_NAV_ITEMS,
  OSS_NAV_ITEMS,
  SettingsNavItem,
} from "#/constants/settings-nav";
import { OrganizationUserRole } from "#/types/org";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { isSettingsPageHidden } from "#/utils/settings-utils";
import { useMe } from "./query/use-me";
import { usePermission } from "./organizations/use-permissions";
import { useOrgTypeAndAccess } from "./use-org-type-and-access";

/**
 * Build Settings navigation items based on:
 * - app mode (saas / oss)
 * - feature flags
 * - active user's role
 * - org type (personal vs team)
 * @returns Settings Nav Items []
 */
export function useSettingsNavItems(): SettingsNavItem[] {
  const { data: config } = useConfig();
  const { data: user } = useMe();
  const userRole: OrganizationUserRole = user?.role ?? "member";
  const { hasPermission } = usePermission(userRole);
  const { isPersonalOrg, isTeamOrg, organizationId } = useOrgTypeAndAccess();

  const shouldHideBilling = isBillingHidden(
    config,
    hasPermission("view_billing"),
  );
  const isSaasMode = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  let items = isSaasMode ? SAAS_NAV_ITEMS : OSS_NAV_ITEMS;

  // First apply feature flag-based hiding
  items = items.filter((item) => !isSettingsPageHidden(item.to, featureFlags));

  // Hide billing when billing is not accessible OR when in team org
  if (shouldHideBilling || isTeamOrg) {
    items = items.filter((item) => item.to !== "/settings/billing");
  }

  // Hide org routes for personal orgs, missing permissions, or no org selected
  if (!hasPermission("view_billing") || !organizationId || isPersonalOrg) {
    items = items.filter((item) => item.to !== "/settings/org");
  }

  if (
    !hasPermission("invite_user_to_organization") ||
    !organizationId ||
    isPersonalOrg
  ) {
    items = items.filter((item) => item.to !== "/settings/org-members");
  }

  return items;
}
