import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

/** Name the agent profile a behavior page edits, or say it can't edit an ACP one. */
export function ActiveProfileScopeNotice({
  profileName,
  isAcp,
  className,
}: {
  profileName: string | null;
  isAcp: boolean;
  className?: string;
}) {
  const { t } = useTranslation("openhands");
  if (!profileName) return null;
  return (
    <Typography.Text
      testId="active-profile-scope-notice"
      className={cn(
        "text-xs",
        isAcp ? "text-warning" : "text-tertiary-alt",
        className,
      )}
    >
      {t(
        isAcp
          ? I18nKey.SETTINGS$BEHAVIOR_ACP_PROFILE
          : I18nKey.SETTINGS$BEHAVIOR_ACTIVE_PROFILE,
        { name: profileName },
      )}
    </Typography.Text>
  );
}

/** Say the profile a behavior page edits could not be loaded, with a retry. */
export function ActiveProfileLoadError({
  onRetry,
  className,
}: {
  onRetry: () => void;
  className?: string;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="active-profile-load-error"
      className={cn("flex flex-col items-start gap-2", className)}
    >
      <Typography.Text className="text-xs text-feedback-error">
        {t(I18nKey.SETTINGS$BEHAVIOR_PROFILE_LOAD_FAILED)}
      </Typography.Text>
      <BrandButton
        testId="active-profile-load-retry"
        type="button"
        variant="secondary"
        onClick={onRetry}
      >
        {t(I18nKey.SETTINGS$BEHAVIOR_PROFILE_RETRY)}
      </BrandButton>
    </div>
  );
}
