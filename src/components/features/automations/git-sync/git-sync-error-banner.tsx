import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { formatTimeDelta } from "#/utils/format-time-delta";

interface GitSyncErrorBannerProps {
  error: string;
  errorAt: string | null;
}

export function GitSyncErrorBanner({
  error,
  errorAt,
}: GitSyncErrorBannerProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      role="alert"
      data-testid="git-sync-error-banner"
      className="rounded-md border border-danger/40 bg-danger/5 p-3 text-sm text-feedback-error whitespace-pre-wrap break-words"
    >
      <p className="font-medium">
        {t(I18nKey.AUTOMATIONS$GIT_SYNC$LAST_ERROR_TITLE)}
      </p>
      <p className="mt-1">{error}</p>
      {errorAt && (
        <p className="mt-1 text-xs">
          {`${formatTimeDelta(errorAt)} ${t(I18nKey.CONVERSATION$AGO)}`}
        </p>
      )}
    </div>
  );
}
