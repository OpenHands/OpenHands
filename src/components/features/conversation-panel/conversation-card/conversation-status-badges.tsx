import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import CircleErrorIcon from "#/icons/circle-error.svg?react";

/** Error sandbox pill beside the conversation title (archived uses the status column icon only). */
export function ConversationStatusBadges() {
  const { t } = useTranslation("openhands");

  return (
    <span
      data-testid="error-badge"
      className="flex items-center gap-1 px-1.5 py-0.5 bg-[var(--oh-status-error)] text-white text-xs font-medium rounded-full"
    >
      <CircleErrorIcon className="text-white w-3 h-3" />
      <span>{t(I18nKey.COMMON$ERROR)}</span>
    </span>
  );
}
