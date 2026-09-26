import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { statusToneBannerClassName } from "#/utils/status-tone-classes";
import { cn } from "#/utils/utils";

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
      className={cn(
        "rounded-md p-3 text-sm whitespace-pre-wrap break-words",
        statusToneBannerClassName.danger,
      )}
    >
      <p className="font-medium">
        {t(I18nKey.AUTOMATIONS$GIT_SYNC$LAST_ERROR_TITLE)}
      </p>
      <p className="mt-1">{error}</p>
      {errorAt && (
        <p className="mt-1 text-xs text-muted">
          {`${formatTimeDelta(errorAt)} ${t(I18nKey.CONVERSATION$AGO)}`}
        </p>
      )}
    </div>
  );
}
