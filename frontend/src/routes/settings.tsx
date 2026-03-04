import { useMemo } from "react";
import { Outlet, redirect, useLocation } from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import {
  WebClientConfig,
  WebClientFeatureFlags,
} from "#/api/option-service/option.types";
import { SettingsLayout } from "#/components/features/settings/settings-layout";
import { Typography } from "#/ui/typography";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";

const SAAS_ONLY_PATHS = [
  "/settings/user",
  "/settings/billing",
  "/settings/credits",
  "/settings/api-keys",
];

/**
 * Checks if a settings page should be hidden based on feature flags.
 * Used by both the route loader and navigation hook to keep logic in sync.
 */
export function isSettingsPageHidden(
  path: string,
  featureFlags: WebClientFeatureFlags | undefined,
): boolean {
  if (featureFlags?.hide_llm_settings && path === "/settings") return true;
  if (featureFlags?.hide_users_page && path === "/settings/user") return true;
  if (featureFlags?.hide_billing_page && path === "/settings/billing")
    return true;
  if (featureFlags?.hide_integrations_page && path === "/settings/integrations")
    return true;
  return false;
}

/**
 * Find the first available settings page that is not hidden.
 * Returns null if no page is available (shouldn't happen in practice).
 */
export function getFirstAvailablePath(
  isSaas: boolean,
  featureFlags: WebClientFeatureFlags | undefined,
): string | null {
  const saasFallbackOrder = [
    { path: "/settings/user", hidden: !!featureFlags?.hide_users_page },
    {
      path: "/settings/integrations",
      hidden: !!featureFlags?.hide_integrations_page,
    },
    { path: "/settings/app", hidden: false },
    { path: "/settings", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings/billing", hidden: !!featureFlags?.hide_billing_page },
    { path: "/settings/secrets", hidden: false },
    { path: "/settings/api-keys", hidden: false },
    { path: "/settings/mcp", hidden: false },
  ];

  const ossFallbackOrder = [
    { path: "/settings", hidden: !!featureFlags?.hide_llm_settings },
    { path: "/settings/mcp", hidden: false },
    {
      path: "/settings/integrations",
      hidden: !!featureFlags?.hide_integrations_page,
    },
    { path: "/settings/app", hidden: false },
    { path: "/settings/secrets", hidden: false },
  ];

  const fallbackOrder = isSaas ? saasFallbackOrder : ossFallbackOrder;
  const firstAvailable = fallbackOrder.find((item) => !item.hidden);

  return firstAvailable?.path ?? null;
}

export const clientLoader = async ({ request }: Route.ClientLoaderArgs) => {
  const url = new URL(request.url);
  const { pathname } = url;

  let config = queryClient.getQueryData<WebClientConfig>(["web-client-config"]);
  if (!config) {
    config = await OptionService.getConfig();
    queryClient.setQueryData<WebClientConfig>(["web-client-config"], config);
  }

  const isSaas = config?.app_mode === "saas";
  const featureFlags = config?.feature_flags;

  // Check if current page should be hidden and redirect to first available page
  const isHiddenPage =
    (!isSaas && SAAS_ONLY_PATHS.includes(pathname)) ||
    isSettingsPageHidden(pathname, featureFlags);

  if (isHiddenPage) {
    const fallbackPath = getFirstAvailablePath(isSaas, featureFlags);
    if (fallbackPath && fallbackPath !== pathname) {
      return redirect(fallbackPath);
    }
    // If no fallback available or same as current, stay on current page
  }

  return null;
};

function SettingsScreen() {
  const { t } = useTranslation();
  const location = useLocation();
  const navItems = useSettingsNavItems();
  // Current section title for the main content area
  const currentSectionTitle = useMemo(() => {
    const currentItem = navItems.find((item) => item.to === location.pathname);
    // Default to the first available navigation item if current page is not found
    return currentItem
      ? currentItem.text
      : (navItems[0]?.text ?? "SETTINGS$TITLE");
  }, [navItems, location.pathname]);

  return (
    <main data-testid="settings-screen" className="h-full">
      <SettingsLayout navigationItems={navItems}>
        <div className="flex flex-col gap-6 h-full">
          <Typography.H2>{t(currentSectionTitle)}</Typography.H2>
          <div className="flex-1 overflow-auto custom-scrollbar-always">
            <Outlet />
          </div>
        </div>
      </SettingsLayout>
    </main>
  );
}

export default SettingsScreen;
