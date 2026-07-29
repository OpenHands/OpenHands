import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { SidebarNavLink } from "#/components/features/sidebar/sidebar-nav-link";
import { AgentCanvasUpdateCard } from "#/components/features/settings/agent-canvas-update-card";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { CloudSettingsLink } from "#/components/features/settings/cloud-settings-link";
import { IntegrationsSettingsLink } from "#/components/features/settings/integrations-settings-link";

interface SettingsDesktopSidebarProps {
  navigationItems: SettingsNavRenderedItem[];
}

/**
 * Desktop sidebar — sibling of the scrolling main column (same pattern as
 * {@link ExtensionsNavigation}). Mobile drawer stays `position: fixed` outside
 * this row in the layout.
 */
export function SettingsDesktopSidebar({
  navigationItems,
}: SettingsDesktopSidebarProps) {
  const { t } = useTranslation("openhands");
  const desktopNavItems = navigationItems.filter(
    (item): item is Extract<SettingsNavRenderedItem, { type: "item" }> =>
      item.type === "item",
  );

  return (
    <aside
      data-testid="settings-navbar-desktop"
      className={cn(
        "hidden md:flex md:w-[260px] md:shrink-0 md:flex-col md:gap-2",
        "md:sticky md:top-8 md:self-start md:pl-8",
      )}
    >
      <Typography.Text className="px-2 text-sm font-normal text-white">
        {t(I18nKey.SETTINGS$TITLE)}
      </Typography.Text>
      <div className="flex flex-col gap-0.5 pt-0.5">
        {desktopNavItems.map((renderedItem) => (
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
      </div>
      <div className="px-2 pt-3">
        <AgentCanvasUpdateCard />
      </div>
      <div className="px-2 pt-3">
        <BackendSyncedSettingsBadge />
      </div>
    </aside>
  );
}
