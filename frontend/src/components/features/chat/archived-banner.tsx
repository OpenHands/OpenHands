import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function ArchivedBanner() {
  const { t } = useTranslation();

  return (
    <div
      data-testid="archived-banner"
      className="flex items-center justify-center px-4 py-3 rounded-lg bg-neutral-700 border border-neutral-600"
    >
      <span className="text-sm text-neutral-300">
        {t(I18nKey.CONVERSATION$ARCHIVED_READ_ONLY)}
      </span>
    </div>
  );
}
