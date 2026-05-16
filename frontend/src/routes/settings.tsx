import { useMemo } from "react";
import { Outlet, redirect, useLocation, useMatches } from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { SettingsLayout } from "#/components/features/settings";
import { WebClientConfig } from "#/api/option-service/option.types";
import {
  QUERY_KEYS,
  CONFIG_CACHE_OPTIONS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { Organization } from "#/types/org";
import { Typography } from "#/ui/typography";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { getSettingsQueryFn } from "#/hooks/query/use-settings";
import { getActiveOrganizationUser } from "#/utils/org/permission-checks";
import { getSelectedOrganizationIdFromStore } from "#/stores/selected-organization-store";
import { rolePermissions } from "#/utils/org/permissions";
import { isBillingHidden } from "#/utils/org/billing-visibility";
import {
  isSettingsPageHidden,
  getFirstAvailablePath,
} from "#/utils/settings-utils";
import { useOrgTypeAndAccess } from "#/hooks/use-org-type-and-access";
import { useConfig } from "#/hooks/query/use-config";
import { useMe } from "#/hooks/query/use-me";
import { OrgWideSettingsBadge } from "#/components/features/settings/org-wide-settings-badge";

const SAAS_ONLY_PATHS = [
  "/settings/user",
  "/settings/billing",
  "/settings/credits",
  "/settings/api-keys",
  "/settings/team",
  "/settings/org",
  "/settings/org-defaults",
  "/settings/org-defaults/condenser",
  "/settings/org-defaults/verification",
];

const ORG_WIDE_BADGE_PATHS = new Set<string>([
  "/settings/org-defaults",
  "/settings/org-defaults/condenser",
  "/settings/org-defaults/verification",
]);

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

  // Step 3b: ACP guard. The LLM / Condenser / MCP personal-settings screens
  // have no useful content while an external ACP subprocess is driving
  // conversations (the sub-agent owns its own tools, LLM, condenser, MCP),
  // so bounce them to ``/settings/agent``. Driven by the nav-item
  // ``disabledByAcp`` flag so this list and the greyed-out nav state
  // ([`use-settings-nav-items.ts`]) come from the same source of truth.
  //
  // Doing the redirect in the loader (instead of a per-route ``useEffect``)
  // means the personal LLM/condenser/MCP pages don't paint a one-frame
  // flash of their content before the guard fires.
  //
  // Gated on ``enable_acp`` so the guard is fully off when the feature
  // flag is disabled. If we can't fetch settings (unauthed, no org, etc.)
  // we fall through and let the page render — same behaviour the
  // previous hook-based guard had.
  if (featureFlags?.enable_acp) {
    const navItems = isSaas ? SAAS_NAV_ITEMS : OSS_NAV_ITEMS;
    const currentItem = navItems.find((item) => item.to === pathname);
    if (currentItem?.disabledByAcp) {
      try {
        const orgId = getSelectedOrganizationIdFromStore();
        const personalSettings = await queryClient.fetchQuery({
          queryKey: SETTINGS_QUERY_KEYS.byScope("personal", orgId),
          queryFn: () => getSettingsQueryFn("personal", orgId),
          staleTime: 1000 * 60 * 5,
        });
        if (personalSettings?.agent_settings?.agent_kind === "acp") {
          return redirect("/settings/agent");
        }
      } catch {
        // Settings unfetchable (unauthed, no org, network) — let the
        // page render rather than redirect-loop on a missing payload.
      }
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
  const { data: config } = useConfig();
  const { isTeamOrg } = useOrgTypeAndAccess();
  const { data: me } = useMe();

  // Determine if we should show the org-wide settings badge
  const isOrgWideBadgePath = ORG_WIDE_BADGE_PATHS.has(location.pathname);
  const isSaasMode = config?.app_mode === "saas";
  const shouldShowOrgWideBadge = isOrgWideBadgePath && isTeamOrg && isSaasMode;
  // Members see a read-only message; Admins/Owners see the org-wide notice.
  const orgWideBadgeVariant =
    me?.role === "member" ? "managed-by-admin" : "org-wide";

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
              {shouldShowOrgWideBadge && (
                <OrgWideSettingsBadge variant={orgWideBadgeVariant} />
              )}
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
