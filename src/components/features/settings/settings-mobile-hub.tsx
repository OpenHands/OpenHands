import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { SidebarNavLink } from "#/components/features/sidebar/sidebar-nav-link";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { CloudSettingsLink } from "#/components/features/settings/cloud-settings-link";
import { IntegrationsSettingsLink } from "#/components/features/settings/integrations-settings-link";

interface SettingsMobileHubProps {
  navigationItems: SettingsNavRenderedItem[];
}

export function SettingsMobileHub({ navigationItems }: SettingsMobileHubProps) {
  const { t } = useTranslation("openhands");

  const navItems = navigationItems.filter(
    (item): item is Extract<SettingsNavRenderedItem, { type: "item" }> =>
      item.type === "item",
  );

  return (
    <div
      data-testid="settings-mobile-hub"
      className="flex flex-col gap-4 px-4 py-2 md:hidden"
    >
      <Typography.H2>{t(I18nKey.SETTINGS$TITLE)}</Typography.H2>
      <nav className="flex flex-col gap-0.5">
        {navItems.map((renderedItem) => (
          <SidebarNavLink
            key={renderedItem.item.to}
            to={renderedItem.item.to}
            label={t(renderedItem.item.text as I18nKey)}
            end
            testId={`sidebar-settings-${renderedItem.item.to}`}
            icon={renderedItem.item.icon}
          />
        ))}
        <IntegrationsSettingsLink />
        <CloudSettingsLink />
      </nav>
      <div className="pt-1">
        <BackendSyncedSettingsBadge />
      </div>
    </div>
  );
}
