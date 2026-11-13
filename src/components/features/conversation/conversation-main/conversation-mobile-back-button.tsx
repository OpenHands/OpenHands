import { ChevronLeft } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import {
  mobileTopBarIconButtonClassName,
  mobileTopBarIconClassName,
} from "#/utils/mobile-top-bar-icon-button-classes";
import { cn } from "#/utils/utils";

const CONVERSATIONS_PATH = "/conversations";

export function ConversationMobileBackButton() {
  const { navigate, currentSearch = "" } = useNavigation();
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid="conversation-mobile-back-button"
      onClick={() => navigate(`${CONVERSATIONS_PATH}${currentSearch}`)}
      aria-label={t(I18nKey.COMMON$BACK)}
      className={cn(
        mobileTopBarIconButtonClassName,
        "size-12 p-0 text-contrast",
      )}
    >
      <ChevronLeft
        className={mobileTopBarIconClassName}
        aria-hidden
        strokeWidth={2}
      />
    </button>
  );
}
