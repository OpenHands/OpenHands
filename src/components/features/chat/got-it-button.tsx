import { useTranslation } from "react-i18next";
import CheckCircle from "#/icons/check-circle-solid.svg?react";
import { I18nKey } from "#/i18n/declaration";

export function GotItButton({ onClick }: { onClick: () => void }) {
  const { t } = useTranslation("openhands");
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-normal text-success bg-success/10 hover:bg-success/20 border border-success/30 transition-colors"
    >
      <CheckCircle className="w-3.5 h-3.5 fill-success" />
      <span>{t(I18nKey.CHAT_INTERFACE$BTW_GOT_IT)}</span>
    </button>
  );
}
