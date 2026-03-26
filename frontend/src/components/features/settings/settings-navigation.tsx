import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import SettingsIcon from "#/icons/settings-gear.svg?react";
import CloseIcon from "#/icons/close.svg?react";
import { OrgSelector } from "../org/org-selector";
import { SettingsNavRenderedItem } from "#/hooks/use-settings-nav-items";
import { useShouldHideOrgSelector } from "#/hooks/use-should-hide-org-selector";
import { SettingsNavHeader } from "./settings-nav-header";
import { SettingsNavDivider } from "./settings-nav-divider";
import { SettingsNavLink } from "./settings-nav-link";

interface SettingsNavigationProps {
  isMobileMenuOpen: boolean;
  onCloseMobileMenu: () => void;
  navigationItems: SettingsNavRenderedItem[];
}

export function SettingsNavigation({
  isMobileMenuOpen,
  onCloseMobileMenu,
  navigationItems,
}: SettingsNavigationProps) {
  const { t } = useTranslation();
  const shouldHideSelector = useShouldHideOrgSelector();

  return (
    <>
      {/* Mobile backdrop */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={onCloseMobileMenu}
        />
      )}
      {/* Navigation sidebar */}
      <nav
        data-testid="settings-navbar"
        className={cn(
          "flex flex-col gap-6 transition-transform duration-300 ease-in-out",
          // Mobile: full screen overlay with dark background matching desktop
          "fixed inset-0 z-50 w-full bg-[#050505] p-4 transform md:transform-none",
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full",
          // Desktop: static sidebar
          "md:relative md:translate-x-0 md:w-64 md:p-0 md:bg-transparent",
        )}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 ml-1 sm:ml-4.5">
            <SettingsIcon width={16} height={16} />
            <Typography.H2>{t(I18nKey.SETTINGS$TITLE)}</Typography.H2>
          </div>
          {/* Close button - only visible on mobile */}
          <button
            type="button"
            onClick={onCloseMobileMenu}
            className="md:hidden p-0.5 hover:bg-tertiary rounded-md transition-colors cursor-pointer"
            aria-label="Close navigation menu"
          >
            <CloseIcon width={32} height={32} />
          </button>
        </div>

        {!shouldHideSelector && <OrgSelector />}

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
        </div>
      </nav>
    </>
  );
}
