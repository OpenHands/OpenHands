import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { SettingsDesktopSidebar } from "./settings-desktop-sidebar";
import { SettingsMobileDrawer } from "./settings-mobile-drawer";

interface SettingsNavigationProps {
  isMobileMenuOpen: boolean;
  onCloseMobileMenu: () => void;
  navigationItems: SettingsNavRenderedItem[];
}

export function SettingsNavigation(props: SettingsNavigationProps) {
  return (
    <>
      <SettingsDesktopSidebar navigationItems={props.navigationItems} />
      <SettingsMobileDrawer {...props} />
    </>
  );
}
