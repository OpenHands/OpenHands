import { useMemo, useState } from "react";
import { Outlet, redirect, useLocation, useMatches } from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { SettingsLayout } from "#/components/features/settings";
import { WebClientConfig } from "#/api/option-service/option.types";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "#/hooks/query/query-keys";
import { Typography } from "#/ui/typography";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import {
  getFirstAvailablePath,
  isSettingsPageHidden,
} from "#/utils/settings-utils";
import { SettingsSectionHeaderProvider } from "#/contexts/settings-section-header-context";

export const clientLoader = async ({ request }: Route.ClientLoaderArgs) => {
  const url = new URL(request.url);
  const { pathname } = url;

  const config = await queryClient.fetchQuery<WebClientConfig>({
    queryKey: QUERY_KEYS.WEB_CLIENT_CONFIG,
    queryFn: OptionService.getConfig,
    ...CONFIG_CACHE_OPTIONS,
  });

  const featureFlags = config?.feature_flags;

  if (isSettingsPageHidden(pathname, featureFlags)) {
    const fallbackPath = getFirstAvailablePath(featureFlags);
    if (fallbackPath && fallbackPath !== pathname) {
      return redirect(fallbackPath);
    }
  }

  return null;
};

function SettingsScreen() {
  const { t } = useTranslation("openhands");
  const location = useLocation();
  const matches = useMatches();
  const navItems = useSettingsNavItems();
  const isMobile = useBreakpoint(768);
  const [hideSectionHeader, setHideSectionHeader] = useState(false);

  const { currentSectionTitle, currentSectionSubtitle } = useMemo(() => {
    const currentRenderedItem = navItems.find(
      (item) => item.type === "item" && item.item.to === location.pathname,
    );
    if (currentRenderedItem?.type === "item") {
      return {
        currentSectionTitle: currentRenderedItem.item.text,
        currentSectionSubtitle: currentRenderedItem.item.subtitle,
      };
    }
    const firstItem = navItems.find((item) => item.type === "item");
    if (firstItem?.type === "item") {
      return {
        currentSectionTitle: firstItem.item.text,
        currentSectionSubtitle: firstItem.item.subtitle,
      };
    }
    return {
      currentSectionTitle: "SETTINGS$TITLE",
      currentSectionSubtitle: null as string | null,
    };
  }, [navItems, location.pathname]);

  const routeHandle = matches.find((m) => m.pathname === location.pathname)
    ?.handle as { hideTitle?: boolean } | undefined;
  const isMobileHub = isMobile && location.pathname === "/settings";
  const shouldHideTitle =
    routeHandle?.hideTitle === true || isMobileHub || hideSectionHeader;

  return (
    <main data-testid="settings-screen" className="min-h-0">
      <SettingsSectionHeaderProvider
        setHideSectionHeader={setHideSectionHeader}
      >
        <SettingsLayout navigationItems={navItems}>
          <div className="flex flex-col gap-6 pb-8">
            {!shouldHideTitle && (
              <header className="space-y-1">
                <Typography.H2>{t(currentSectionTitle)}</Typography.H2>
                {currentSectionSubtitle ? (
                  <p
                    data-testid="settings-page-subtitle"
                    className="text-sm leading-5 text-tertiary-light"
                  >
                    {t(currentSectionSubtitle)}
                  </p>
                ) : null}
              </header>
            )}
            <Outlet />
          </div>
        </SettingsLayout>
      </SettingsSectionHeaderProvider>
    </main>
  );
}

export default SettingsScreen;
