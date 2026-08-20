import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { Pre } from "#/ui/pre";

interface SkillContentProps {
  content: string;
}

export function SkillContent({ content }: SkillContentProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="mt-2">
      <Typography.Text className="text-sm font-semibold text-[var(--oh-text-tertiary)] mb-2">
        {t(I18nKey.COMMON$CONTENT)}
      </Typography.Text>
      <Pre
        size="small"
        font="mono"
        lineHeight="relaxed"
        padding="medium"
        borderRadius="medium"
        maxHeight="small"
        overflow="auto"
        className="mt-2 border border-[var(--oh-border)] bg-base text-[var(--oh-text-tertiary)]"
      >
        {content || t(I18nKey.SKILLS_MODAL$NO_CONTENT)}
      </Pre>
    </div>
  );
}
