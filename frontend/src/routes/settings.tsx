import { useMemo } from "react";
import { Outlet, redirect, useLocation, useMatches } from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { SettingsLayout } from "#/components/features/settings";
import { WebClientConfig } from "#/api/option-service/option.types";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "#/hooks/query/query-keys";
import { Organization } from "#/types/org";
import { Typography } from "#/ui/typography";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { getActiveOrganizationUser } from "#/utils/org/permission-checks";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { rolePermissions } from "#/utils/org/permissions";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import {
  isSettingsPageHidden,
  getFirstAvailablePath,
} from "#/utils/settings-utils";
import { useMe } from "#/hooks/query/use-me";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { useConfig } from "#/hooks/query/use-config";
import { OrgWideSettingsBadge } from "#/components/features/settings/org-wide-settings-badge";

const SAAS_ONLY_PATHS = [
  "/settings/user",
  "/settings/billing",
  "/settings/credits",
  "/settings/api-keys",
  "/settings/team",
  "/settings/org",
];

export const clientLoader = async ({ request }: Route.ClientLoaderArgs) => {
  const url = new URL(request.url);
  const { pathname } = url;

  // Step 1: Get config first (needed for all checks, no user data required)
  const config = await queryClient.fetchQuery<WebClientConfig>({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
  });

  const isSaas = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  // Step 2: Check SAAS_ONLY_PATHS for OSS mode (no user data required)
  if (!isSaas && SAAS_ONLY_PATHS.includes(pathname)) {
    return redirect("/settings");
  }

  // Step 3: Check feature flag-based hiding and redirect IMMEDIATELY (no user data required)
  // This handles hide_llm_settings, hide_users_page, hide_billing_page, hide_integrations_page
  if (isSettingsPageHidden(pathname, featureFlags)) {
    const fallbackPath = getFirstAvailablePath(isSaas, featureFlags);
    if (fallbackPath && fallbackPath !== pathname) {
      return redirect(fallbackPath);
    }
  }

  // Step 4: For routes that need permission checks, get user data
  // Only fetch user data for billing and org routes that need permission validation
  if (
    pathname === "/settings/billing" ||
    pathname === "/settings/org" ||
    pathname === "/settings/org-members"
  ) {
    const user = await getActiveOrganizationUser();

    // Org-type detection for route protection
    const orgId = getSelectedOrganizationIdFromStore();
    const organizationsData = queryClient.getQueryData<{
      items: Organization[];
      currentOrgId: string | null;
    }>(["organizations"]);
    const selectedOrg = organizationsData?.items?.find(
      (org) => org.id === orgId,
    );
    const isPersonalOrg = selectedOrg?.is_personal === true;
    const isTeamOrg = !!selectedOrg && !selectedOrg.is_personal;

    // Billing route protection
    if (pathname === "/settings/billing") {
      if (
        !user ||
        isBillingHidden(
          config,
          rolePermissions[user.role ?? "member"].includes("view_billing"),
        ) ||
        isTeamOrg
      ) {
        if (isSaas) {
          const fallbackPath = getFirstAvailablePath(isSaas, featureFlags);
          return redirect(fallbackPath ?? "/settings");
        }
      }
    }

    // Org route protection: redirect if user lacks required permissions or personal org
    if (pathname === "/settings/org" || pathname === "/settings/org-members") {
      const role = user?.role ?? "member";
      const requiredPermission =
        pathname === "/settings/org"
          ? "view_billing"
          : "invite_user_to_organization";

      if (
        !user ||
        !rolePermissions[role].includes(requiredPermission) ||
        isPersonalOrg
      ) {
        return redirect("/settings");
      }
    }
  }

  return null;
};

function SettingsScreen() {
  const { t } = useTranslation();
  const location = useLocation();
  const matches = useMatches();
  const navItems = useSettingsNavItems();
  const { data: me } = useMe();
  const { data: config } = useConfig();
  const { isTeamOrg } = useOrgTypeAndAccess();

  // Determine if we should show the org-wide settings badge
  // Only show for Admin/Owner roles on the LLM settings page in team orgs
  const isLlmSettingsPage = location.pathname === "/settings";
  const isAdminOrOwner = me?.role === "admin" || me?.role === "owner";
  const isSaasMode = config?.app_mode === "saas";
  const shouldShowOrgWideBadge =
    isLlmSettingsPage && isAdminOrOwner && isTeamOrg && isSaasMode;

  // Current section title for the main content area
  const currentSectionTitle = useMemo(() => {
    // Find the current item from rendered items
    const currentRenderedItem = navItems.find(
      (item) => item.type === "item" && item.item.to === location.pathname,
    );
    if (currentRenderedItem && currentRenderedItem.type === "item") {
      return currentRenderedItem.item.text;
    }
    // Default to the first available navigation item if current page is not found
    const firstItem = navItems.find((item) => item.type === "item");
    return firstItem && firstItem.type === "item"
      ? firstItem.item.text
      : "SETTINGS$TITLE";
  }, [navItems, location.pathname]);

  const routeHandle = matches.find((m) => m.pathname === location.pathname)
    ?.handle as { hideTitle?: boolean } | undefined;
  const shouldHideTitle = routeHandle?.hideTitle === true;

  return (
    <main data-testid="settings-screen" className="h-full">
      <SettingsLayout navigationItems={navItems}>
        <div className="flex flex-col gap-6 h-full">
          {!shouldHideTitle && (
            <div className="flex items-center gap-3 flex-wrap">
              <Typography.H2>{t(currentSectionTitle)}</Typography.H2>
              {shouldShowOrgWideBadge && <OrgWideSettingsBadge />}
            </div>
          )}
          <div className="flex-1 overflow-auto custom-scrollbar-always">
            <Outlet />
          </div>
        </div>
      </SettingsLayout>
    </main>
  );
}

export default SettingsScreen;
