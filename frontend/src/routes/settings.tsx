import { useMemo } from "react";
import { Outlet, redirect, useLocation } from "react-router";
import { useTranslation } from "react-i18next";
import { useConfig } from "#/hooks/query/use-config";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { GetConfigResponse } from "#/api/option-service/option.types";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { SettingsLayout } from "#/components/features/settings";
import { Typography } from "#/ui/typography";

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

  let config = queryClient.getQueryData<GetConfigResponse>(["config"]);
  if (!config) {
    config = await OptionService.getConfig();
    queryClient.setQueryData<GetConfigResponse>(["config"], config);
  }

  const isSaas = config?.APP_MODE === "saas";

  if (!isSaas && SAAS_ONLY_PATHS.includes(pathname)) {
    // if in OSS mode, do not allow access to saas-only paths
    return redirect("/settings");
  }

  return null;
};

function SettingsScreen() {
  const { t } = useTranslation();
  const { data: config } = useConfig();
  const location = useLocation();

  const isSaas = config?.APP_MODE === "saas";

  // Navigation items configuration
  const navItems = useMemo(() => {
    const items = [];
    if (isSaas) {
      items.push(...SAAS_NAV_ITEMS);
    } else {
      items.push(...OSS_NAV_ITEMS);
    }
    return items;
  }, [isSaas]);

  // Current section title for the main content area
  const currentSectionTitle = useMemo(() => {
    const currentItem = navItems.find((item) => item.to === location.pathname);
    return currentItem ? currentItem.text : "SETTINGS$NAV_LLM";
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
