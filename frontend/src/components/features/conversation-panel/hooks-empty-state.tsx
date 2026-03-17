import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface HooksEmptyStateProps {
  isError: boolean;
}

export function HooksEmptyState({ isError }: HooksEmptyStateProps) {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-center h-full p-4">
      <Typography.Text className="text-gray-400">
        {isError
          ? t(I18nKey.COMMON$FETCH_ERROR)
          : t(I18nKey.CONVERSATION$NO_HOOKS)}
      </Typography.Text>
    </div>
  );
}
