import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface GitLabWebhookManagerStateProps {
  className?: string;
  titleKey: I18nKey;
  messageKey: I18nKey;
  messageColor?: string;
}

export function GitLabWebhookManagerState({
  className,
  titleKey,
  messageKey,
  messageColor = "text-gray-400",
}: GitLabWebhookManagerStateProps) {
  const { t } = useTranslation();

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <Typography.H3 className="text-lg font-medium text-white">
        {t(titleKey)}
      </Typography.H3>
      <Typography.Text className={cn("text-sm", messageColor)}>
        {t(messageKey)}
      </Typography.Text>
    </div>
  );
}
