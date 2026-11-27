import { Navigate } from "react-router";
import {
  SETTINGS_COMPACT_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { SettingsMobileHub } from "#/components/features/settings/settings-mobile-hub";
import { LOCKED_CLOUD_SETTINGS_NAV_PATH } from "#/constants/settings-nav";

export default function SettingsIndex() {
  const isMobile = useBreakpoint(SETTINGS_COMPACT_MAX_WIDTH);
  const navigationItems = useSettingsNavItems();

  if (isMobile) {
    return <SettingsMobileHub navigationItems={navigationItems} />;
  }

  return <Navigate to={LOCKED_CLOUD_SETTINGS_NAV_PATH} replace />;
}
