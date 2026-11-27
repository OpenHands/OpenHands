import { AlertTriangle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function ModelCatalogWarning({
  compact = false,
}: {
  compact?: boolean;
}) {
  const { t } = useTranslation("openhands");
  const explanation = t(I18nKey.SETTINGS$MODEL_NOT_LISTED_HELP);

  return (
    <span
      className="inline-flex shrink-0 items-center gap-1 text-xs text-warning"
      title={explanation}
      aria-label={explanation}
      data-testid="model-catalog-warning"
    >
      <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
      {!compact && t(I18nKey.SETTINGS$MODEL_NOT_LISTED)}
    </span>
  );
}
