import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ExclamationCircleIcon from "#/icons/exclamation-circle.svg?react";

interface ErrorStateProps {
  onRetry: () => void;
}

export function ErrorState({ onRetry }: ErrorStateProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <ExclamationCircleIcon className="size-12 text-danger" />
      <p className="mt-4 text-sm text-muted">
        {t(I18nKey.AUTOMATIONS$ERROR_TITLE)}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-4 rounded-lg border border-[var(--oh-border)] px-4 py-2 text-sm text-white hover:bg-surface-raised"
      >
        {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
      </button>
    </div>
  );
}
