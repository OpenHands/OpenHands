import { useMemo } from "react";
import { Outlet, redirect, useLocation, useMatches } from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { SettingsLayout } from "#/components/features/settings";
import { WebClientConfig } from "#/api/option-service/option.types";
import { Organization } from "#/types/org";
import { Typography } from "#/ui/typography";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { getActiveOrganizationUser } from "#/utils/org/permission-checks";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { rolePermissions } from "#/utils/org/permissions";
import { isBillingHidden } from "#/utils/org/billing-visibility";

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
  const user = await getActiveOrganizationUser();

  let config = queryClient.getQueryData<WebClientConfig>(["web-client-config"]);
  if (!config) {
    config = await OptionService.getConfig();
    queryClient.setQueryData<WebClientConfig>(["web-client-config"], config);
  }

  const isSaas = config?.app_mode === "saas";

  if (!isSaas && SAAS_ONLY_PATHS.includes(pathname)) {
    // if in OSS mode, do not allow access to saas-only paths
    return redirect("/settings");
  }
  // If LLM settings are hidden and user tries to access the LLM settings page
  if (config?.feature_flags?.hide_llm_settings && pathname === "/settings") {
    // Redirect to the first available settings page
    return isSaas ? redirect("/settings/user") : redirect("/settings/mcp");
  }

  // Org-type detection for route protection
  const orgId = getSelectedOrganizationIdFromStore();
  const organizationsData = queryClient.getQueryData<{
    items: Organization[];
    currentOrgId: string | null;
  }>(["organizations"]);
  const selectedOrg = organizationsData?.items?.find((org) => org.id === orgId);
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
        return redirect("/settings/user");
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

  return null;
};

function SettingsScreen() {
  const { t } = useTranslation();
  const location = useLocation();
  const matches = useMatches();
  const navItems = useSettingsNavItems();

  // Current section title for the main content area
  const currentSectionTitle = useMemo(() => {
    const currentItem = navItems.find((item) => item.to === location.pathname);
    // Default to the first available navigation item if current page is not found
    return currentItem
      ? currentItem.text
      : (navItems[0]?.text ?? "SETTINGS$TITLE");
  }, [navItems, location.pathname]);

  const routeHandle = matches.find((m) => m.pathname === location.pathname)
    ?.handle as { hideTitle?: boolean } | undefined;
  const shouldHideTitle = routeHandle?.hideTitle === true;

  return (
    <main data-testid="settings-screen" className="h-full">
      <SettingsLayout navigationItems={navItems}>
        <div className="flex flex-col gap-6 h-full">
          {!shouldHideTitle && (
            <Typography.H2>{t(currentSectionTitle)}</Typography.H2>
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
