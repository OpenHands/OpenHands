import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { SidebarNavLink } from "#/components/features/sidebar/sidebar-nav-link";
import { AgentCanvasUpdateCard } from "#/components/features/settings/agent-canvas-update-card";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { CloudSettingsLink } from "#/components/features/settings/cloud-settings-link";
import { IntegrationsSettingsLink } from "#/components/features/settings/integrations-settings-link";
import { SettingsNavHeader } from "#/components/features/settings/settings-nav-header";
import { cn } from "#/utils/utils";

interface SettingsMobileHubProps {
  navigationItems: SettingsNavRenderedItem[];
}

export function SettingsMobileHub({ navigationItems }: SettingsMobileHubProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="settings-mobile-hub"
      className="flex flex-col gap-4 px-4 py-2 lg:hidden"
    >
      <Typography.H2>{t(I18nKey.SETTINGS$TITLE)}</Typography.H2>
      <nav className="flex flex-col gap-0.5">
        {navigationItems.map((renderedItem, index) => {
          if (renderedItem.type === "header") {
            return (
              <SettingsNavHeader
                key={`header-${renderedItem.text}`}
                text={renderedItem.text}
                className={cn("px-2.5 pb-0.5", index > 0 && "pt-3")}
              />
            );
          }
          if (renderedItem.type === "divider") return null;
          return (
            <SidebarNavLink
              key={renderedItem.item.to}
              to={renderedItem.item.to}
              label={t(renderedItem.item.text as I18nKey)}
              description={t(renderedItem.item.subtitle as I18nKey)}
              end
              testId={`sidebar-settings-${renderedItem.item.to}`}
              icon={renderedItem.item.icon}
            />
          );
        })}
        <IntegrationsSettingsLink />
        <CloudSettingsLink />
      </nav>
      <AgentCanvasUpdateCard />
      <div className="pt-1">
        <BackendSyncedSettingsBadge />
      </div>
    </div>
  );
}
