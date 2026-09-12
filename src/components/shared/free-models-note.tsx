import { Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function FreeOpenHandsModelsNote({
  modelIds,
}: {
  modelIds: Iterable<string>;
}) {
  const { t } = useTranslation("openhands");

  return (
    <p
      data-testid="openhands-free-models-note"
      className="flex items-start gap-2 text-xs text-warning"
    >
      <Info className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
      <span>
        {t(I18nKey.SETTINGS$OPENHANDS_FREE_MODELS_NOTE, {
          ids: [...modelIds].join(", "),
        })}
      </span>
    </p>
  );
}
