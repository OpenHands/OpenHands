import { useTranslation } from "react-i18next";
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
