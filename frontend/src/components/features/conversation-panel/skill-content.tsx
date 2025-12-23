import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { Pre } from "#/ui/pre";

interface SkillContentProps {
  content: string;
}

export function SkillContent({ content }: SkillContentProps) {
  const { t } = useTranslation();

  return (
    <div className="mt-2">
      <Typography.Text className="text-sm font-semibold text-gray-300 mb-2">
        {t(I18nKey.COMMON$CONTENT)}
      </Typography.Text>
      <Pre
        size="default"
        font="mono"
        lineHeight="relaxed"
        background="dark"
        textColor="light"
        padding="medium"
        borderRadius="medium"
        shadow="inner"
        maxHeight="small"
        overflow="auto"
        className="mt-2"
      >
        {content || t(I18nKey.SKILLS_MODAL$NO_CONTENT)}
      </Pre>
    </div>
  );
}
