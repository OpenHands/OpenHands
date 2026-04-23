import { useConfig } from "#/hooks/query/use-config";
import {
  SAAS_NAV_ITEMS,
  OSS_NAV_ITEMS,
  SettingsNavItem,
  SettingsNavSection,
} from "#/constants/settings-nav";
import { OrganizationUserRole } from "#/types/org";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import { isSettingsPageHidden } from "#/utils/settings-utils";
import { useMe } from "./query/use-me";
import { usePermission } from "./organizations/use-permissions";
import { useOrgTypeAndAccess } from "./use-org-type-and-access";
import { I18nKey } from "#/i18n/declaration";

// Rendered navigation item types
export type SettingsNavRenderedItem =
  | { type: "item"; item: SettingsNavItem }
  | { type: "header"; text: I18nKey }
  | { type: "divider" };

// Section header text mapping
const SECTION_HEADERS: Partial<Record<SettingsNavSection, I18nKey>> = {
  org: I18nKey.SETTINGS$ORG_SETTINGS_HEADER,
  personal: I18nKey.SETTINGS$PERSONAL_SETTINGS_HEADER,
};

/**
 * Build Settings navigation items based on:
 * - app mode (saas / oss)
 * - feature flags
 * - active user's role
 * - org type (personal vs team)
 * @returns Settings Nav Rendered Items (items, headers, dividers)
 */
export function useSettingsNavItems(): SettingsNavRenderedItem[] {
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
  const isAdminOrOwner = userRole === "admin" || userRole === "owner";

  let items = isSaasMode ? [...SAAS_NAV_ITEMS] : [...OSS_NAV_ITEMS];

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

  if (!organizationId) {
    items = items.filter(
      (item) => !item.to.startsWith("/settings/org-defaults"),
    );
  }

  const PERSONAL_LLM_PATHS = new Set([
    "/settings",
    "/settings/condenser",
    "/settings/verification",
  ]);
  if (isSaasMode) {
    items = items.filter((item) => !PERSONAL_LLM_PATHS.has(item.to));
  }

  // For OSS mode or non-SaaS, return flat list without sections
  if (!isSaasMode) {
    return items.map((item) => ({ type: "item", item }));
  }

  // Build rendered items with headers and dividers for SaaS mode
  const renderedItems: SettingsNavRenderedItem[] = [];
  let currentSection: SettingsNavSection | undefined;
  let isFirstSection = true;

  // Determine if we should show section headers (only for admins/owners in team orgs)
  const showSectionHeaders = isTeamOrg && isAdminOrOwner;

  for (const item of items) {
    const itemSection = item.section;

    // Check if we're entering a new section
    if (itemSection && itemSection !== currentSection) {
      // For personal orgs or members, treat "org" and "personal" sections as one group
      // (LLM is the only org item visible and should flow with personal items)
      const isOrgToPersonalWithoutHeaders =
        (isPersonalOrg || !isAdminOrOwner) &&
        currentSection === "org" &&
        itemSection === "personal";

      // Add divider between sections (but not before the first section,
      // and not between org->personal when section headers aren't shown)
      if (!isFirstSection && !isOrgToPersonalWithoutHeaders) {
        renderedItems.push({ type: "divider" });
      }

      // Add section header for org and personal sections (admins/owners only)
      if (showSectionHeaders && SECTION_HEADERS[itemSection]) {
        renderedItems.push({
          type: "header",
          text: SECTION_HEADERS[itemSection]!,
        });
      }

      currentSection = itemSection;
      isFirstSection = false;
    }

    renderedItems.push({ type: "item", item });
  }

  return renderedItems;
}
