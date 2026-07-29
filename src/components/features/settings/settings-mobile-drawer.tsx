import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import SettingsIcon from "#/icons/settings-gear.svg?react";
import CloseIcon from "#/icons/close.svg?react";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { SettingsNavHeader } from "./settings-nav-header";
import { SettingsNavDivider } from "./settings-nav-divider";
import { SettingsNavLink } from "./settings-nav-link";
import { navInteractiveTransitionClassName } from "#/components/features/sidebar/sidebar-layout";
import { AgentCanvasUpdateCard } from "#/components/features/settings/agent-canvas-update-card";
import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { CloudSettingsLink } from "#/components/features/settings/cloud-settings-link";
import { IntegrationsSettingsLink } from "#/components/features/settings/integrations-settings-link";

interface SettingsMobileDrawerProps {
  isMobileMenuOpen: boolean;
  onCloseMobileMenu: () => void;
  navigationItems: SettingsNavRenderedItem[];
}

/**
 * Mobile overlay + drawer. Rendered outside the scrolling flex row so `position:
 * fixed` does not interact with flex item sizing on desktop.
 */
export function SettingsMobileDrawer({
  isMobileMenuOpen,
  onCloseMobileMenu,
  navigationItems,
}: SettingsMobileDrawerProps) {
  const { t } = useTranslation("openhands");

  return (
    <>
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black bg-opacity-50 md:hidden"
          onClick={onCloseMobileMenu}
        />
      )}
      <nav
        data-testid="settings-navbar"
        className={cn(
          "flex flex-col gap-6 transition-transform duration-300 ease-in-out",
          "fixed inset-0 z-50 w-full bg-[var(--oh-surface-deep)] p-4 transform md:hidden",
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between">
          <div className="ml-1 flex items-center gap-2 sm:ml-4.5">
            <SettingsIcon width={16} height={16} />
            <Typography.H2>{t(I18nKey.SETTINGS$TITLE)}</Typography.H2>
          </div>
          <button
            type="button"
            onClick={onCloseMobileMenu}
            className={cn(
              "cursor-pointer rounded-md p-0.5 hover:bg-tertiary md:hidden",
              navInteractiveTransitionClassName,
            )}
            aria-label={t(I18nKey.SIDEBAR$CLOSE_MENU)}
          >
            <CloseIcon width={32} height={32} />
          </button>
        </div>

        <div className="flex flex-col gap-2">
          {navigationItems.map((renderedItem, index) => {
            if (renderedItem.type === "header") {
              return (
                <SettingsNavHeader
                  key={`header-${renderedItem.text}`}
                  text={renderedItem.text}
                />
              );
            }

            if (renderedItem.type === "divider") {
              return <SettingsNavDivider key={`divider-${index}`} />;
            }

            return (
              <SettingsNavLink
                key={renderedItem.item.to}
                item={renderedItem.item}
                onClick={onCloseMobileMenu}
              />
            );
          })}
          <IntegrationsSettingsLink />
          <CloudSettingsLink />
        </div>

        <div className="px-2 pt-3">
          <AgentCanvasUpdateCard />
        </div>

        <div className="px-2 pt-3">
          <BackendSyncedSettingsBadge />
        </div>
      </nav>
    </>
  );
}
