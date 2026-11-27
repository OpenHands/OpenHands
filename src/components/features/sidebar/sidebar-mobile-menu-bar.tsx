import { ChevronLeft } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  mobileTopBarIconButtonClassName,
  mobileTopBarIconClassName,
} from "#/utils/mobile-top-bar-icon-button-classes";
import { useNavigation } from "#/context/navigation-context";
import { getMobileTopBarState } from "#/utils/mobile-section-nav";
import { SidebarMobileMenuToggle } from "./sidebar-mobile-menu-toggle";

export function SidebarMobileMenuBar() {
  const { t } = useTranslation("openhands");
  const { currentPath, navigate } = useNavigation();
  const topBar = getMobileTopBarState(currentPath);

  return (
    <header
      className="flex md:hidden h-12 shrink-0 items-center gap-2 px-2.5"
      aria-label={t(I18nKey.SIDEBAR$NAVIGATION_LABEL)}
    >
      <SidebarMobileMenuToggle />
      {topBar.mode === "back" && topBar.backTo ? (
        <button
          type="button"
          data-testid="sidebar-mobile-back-button"
          onClick={() => navigate(topBar.backTo!)}
          aria-label={t(topBar.backLabelKey ?? I18nKey.COMMON$BACK)}
          className={mobileTopBarIconButtonClassName}
        >
          <ChevronLeft
            size={20}
            className={mobileTopBarIconClassName}
            aria-hidden
            strokeWidth={2}
          />
        </button>
      ) : null}
    </header>
  );
}
