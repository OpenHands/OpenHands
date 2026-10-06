import { useMemo, useState } from "react";
import {
  Navigate,
  Outlet,
  redirect,
  useLocation,
  useMatches,
} from "react-router";
import { useTranslation } from "react-i18next";
import { Route } from "./+types/settings";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import { SettingsLayout } from "#/components/features/settings";
import { WebClientConfig } from "#/api/option-service/option.types";
import { QUERY_KEYS, CONFIG_CACHE_OPTIONS } from "#/hooks/query/query-keys";
import { Typography } from "#/ui/typography";
import {
  SETTINGS_COMPACT_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import {
  LOCKED_CLOUD_SETTINGS_NAV_PATH,
  OSS_NAV_ITEMS,
} from "#/constants/settings-nav";
import {
  getFirstAvailablePath,
  isSettingsPageHidden,
} from "#/utils/settings-utils";
import { SettingsSectionHeaderProvider } from "#/contexts/settings-section-header-context";
import { getLockedCloudHost } from "#/api/agent-server-config";

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
  const isMobile = useBreakpoint(SETTINGS_COMPACT_MAX_WIDTH);
  const [hideSectionHeader, setHideSectionHeader] = useState(false);

  const { currentSectionTitle, currentSectionSubtitle } = useMemo(() => {
    // Resolve from the full list, not the listed subset, so a page that is
    // reachable but unlisted still gets its own title.
    const currentItem = OSS_NAV_ITEMS.find(
      (item) => item.to === location.pathname,
    );
    if (currentItem) {
      return {
        currentSectionTitle: currentItem.text,
        currentSectionSubtitle: currentItem.subtitle,
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

  // Locked-to-Cloud (SaaS / self-hosted OHE) only exposes the Application
  // page; the OHE settings shell owns the rest, so direct links to the other
  // Canvas settings pages are sent to Application instead (OHE-3457).
  if (
    getLockedCloudHost() !== null &&
    location.pathname !== "/settings" &&
    location.pathname !== LOCKED_CLOUD_SETTINGS_NAV_PATH
  ) {
    return <Navigate to={LOCKED_CLOUD_SETTINGS_NAV_PATH} replace />;
  }

  // SettingsLayout renders the page's only <main> landmark (#17909).
  return (
    <div data-testid="settings-screen" className="min-h-0">
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
    </div>
  );
}

export default SettingsScreen;
