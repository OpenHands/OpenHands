import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { getApiErrorMessage } from "#/utils/api-error-message";

interface SettingsLoadErrorProps {
  error: unknown;
  onRetry: () => void;
}

/**
 * The explicit failure state of a settings surface. Rendered instead of the
 * editable form whenever the settings load rejects: saving defaults the user
 * never saw is worse than an actionable error.
 */
export function SettingsLoadError({ error, onRetry }: SettingsLoadErrorProps) {
  const { t } = useTranslation("openhands");
  const detail =
    error !== undefined && error !== null ? getApiErrorMessage(error, "") : "";

  return (
    <div className="flex flex-col gap-3" data-testid="settings-load-failed">
      <div className="flex items-center gap-2">
        <Typography.Text className="text-sm text-feedback-error">
          {t(I18nKey.SETTINGS$LOAD_FAILED)}
        </Typography.Text>
        <BrandButton
          testId="settings-load-retry"
          type="button"
          variant="secondary"
          onClick={onRetry}
        >
          {t(I18nKey.SETTINGS$LOAD_RETRY)}
        </BrandButton>
      </div>
      {detail ? (
        <Typography.Text className="text-xs text-muted">
          {detail}
        </Typography.Text>
      ) : null}
    </div>
  );
}
