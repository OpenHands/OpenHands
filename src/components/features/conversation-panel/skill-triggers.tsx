import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface SkillTriggersProps {
  triggers: string[];
}

export function SkillTriggers({ triggers }: SkillTriggersProps) {
  const { t } = useTranslation("openhands");

  if (!triggers || triggers.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 mb-3">
      <Typography.Text className="text-sm font-semibold text-[var(--oh-text-tertiary)] mb-2">
        {t(I18nKey.COMMON$TRIGGERS)}
      </Typography.Text>
      <div className="mt-2 flex flex-wrap gap-1">
        {triggers.map((trigger) => (
          <span
            key={trigger}
            className="inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium leading-4 border border-[var(--oh-border)] bg-[var(--oh-surface)] text-tertiary-light"
          >
            {trigger}
          </span>
        ))}
      </div>
    </div>
  );
}
