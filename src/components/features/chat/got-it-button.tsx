import { useTranslation } from "react-i18next";
import CheckCircle from "#/icons/check-circle-solid.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { statusToneBadgeClassName } from "#/utils/status-tone-classes";
import { cn } from "#/utils/utils";

export function GotItButton({ onClick }: { onClick: () => void }) {
  const { t } = useTranslation("openhands");
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-normal border border-success/30 hover:border-success transition-colors",
        statusToneBadgeClassName.success,
      )}
    >
      <CheckCircle className="w-3.5 h-3.5 fill-current" />
      <span>{t(I18nKey.CHAT_INTERFACE$BTW_GOT_IT)}</span>
    </button>
  );
}
