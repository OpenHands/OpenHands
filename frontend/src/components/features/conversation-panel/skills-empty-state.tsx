import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface SkillsEmptyStateProps {
  isError: boolean;
}

export function SkillsEmptyState({ isError }: SkillsEmptyStateProps) {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-center h-full p-4">
      <Typography.Text className="text-gray-400">
        {isError
          ? t(I18nKey.COMMON$FETCH_ERROR)
          : t(I18nKey.CONVERSATION$NO_SKILLS)}
      </Typography.Text>
    </div>
  );
}
