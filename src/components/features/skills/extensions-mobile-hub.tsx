import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SidebarNavLink } from "#/components/features/sidebar/sidebar-nav-link";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { EXTENSIONS_NAV_ITEMS } from "./extensions-navigation";

export function ExtensionsMobileHub() {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="extensions-mobile-hub"
      className="flex flex-col gap-4 px-4 py-2 lg:hidden"
    >
      <Typography.H2>{t(I18nKey.NAV$CUSTOMIZE)}</Typography.H2>
      <nav className="flex flex-col gap-0.5">
        {EXTENSIONS_NAV_ITEMS.map((item) => (
          <SidebarNavLink
            key={item.to}
            to={item.to}
            label={item.label}
            end={item.end}
            testId={`sidebar-extensions-${item.to}`}
            icon={item.icon}
            disabled={item.comingSoon}
          />
        ))}
      </nav>
      <div className="pt-1">
        <BackendSyncedSettingsBadge />
      </div>
    </div>
  );
}
