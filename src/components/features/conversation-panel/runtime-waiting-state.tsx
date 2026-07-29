import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface RuntimeWaitingStateProps {
  testId?: string;
  messageKey?: I18nKey;
  className?: string;
}

export function RuntimeWaitingState({
  testId = "runtime-waiting",
  messageKey = I18nKey.DIFF_VIEWER$WAITING_FOR_RUNTIME,
  className,
}: RuntimeWaitingStateProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid={testId}
      className={cn(
        "flex h-full w-full flex-col items-center justify-center gap-3 py-8 text-center",
        className,
      )}
    >
      <LoadingSpinner size="small" />
      <Typography.Text className="text-sm text-[var(--oh-muted)]">
        {t(messageKey)}
      </Typography.Text>
    </div>
  );
}
