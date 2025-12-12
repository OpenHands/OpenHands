import { useTranslation } from "react-i18next";
import SettingsIcon from "#/icons/settings-gear.svg?react";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

interface MobileHeaderProps {
  isMobileMenuOpen: boolean;
  onToggleMenu: () => void;
}

export function MobileHeader({
  isMobileMenuOpen,
  onToggleMenu,
}: MobileHeaderProps) {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-between mb-4 md:hidden">
      <div className="flex items-center gap-2">
        <SettingsIcon width={16} height={16} />
        <Typography.H2>{t(I18nKey.SETTINGS$TITLE)}</Typography.H2>
      </div>
      <button
        type="button"
        onClick={onToggleMenu}
        className="p-2 rounded-md bg-tertiary hover:bg-[#454545] transition-colors"
        aria-label="Toggle settings menu"
      >
        <svg
          width={20}
          height={20}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          {isMobileMenuOpen ? (
            <>
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </>
          ) : (
            <>
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </>
          )}
        </svg>
      </button>
    </div>
  );
}
