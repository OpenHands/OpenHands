import { Navigate } from "react-router";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { useConfig } from "#/hooks/query/use-config";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { SettingsMobileHub } from "#/components/features/settings/settings-mobile-hub";
import { getFirstAvailablePath } from "#/utils/settings-utils";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { LOCKED_CLOUD_SETTINGS_NAV_PATH } from "#/constants/settings-nav";

export default function SettingsIndex() {
  const isMobile = useBreakpoint(768);
  const navigationItems = useSettingsNavItems();
  const { data: config } = useConfig();

  if (isMobile) {
    return <SettingsMobileHub navigationItems={navigationItems} />;
  }

  // Locked-to-Cloud (SaaS / self-hosted OHE) lists only the Application page,
  // so land there instead of the Agent library (OHE-3168).
  const fallbackPath =
    getLockedCloudHost() !== null
      ? LOCKED_CLOUD_SETTINGS_NAV_PATH
      : (getFirstAvailablePath(config?.feature_flags) ?? "/settings/app");

  return <Navigate to={fallbackPath} replace />;
}
