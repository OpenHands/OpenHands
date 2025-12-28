import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface SkillTriggersProps {
  triggers: string[];
}

export function SkillTriggers({ triggers }: SkillTriggersProps) {
  const { t } = useTranslation();

  if (!triggers || triggers.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 mb-3">
      <Typography.Text className="text-sm font-semibold text-gray-300 mb-2">
        {t(I18nKey.COMMON$TRIGGERS)}
      </Typography.Text>
      <div className="flex flex-wrap gap-1">
        {triggers.map((trigger) => (
          <Typography.Text
            key={trigger}
            className="px-2 py-1 text-xs rounded-full bg-blue-900"
          >
            {trigger}
          </Typography.Text>
        ))}
      </div>
    </div>
  );
}
