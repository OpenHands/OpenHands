import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface SettingsNavHeaderProps {
  text: I18nKey;
  className?: string;
}

export function SettingsNavHeader({ text, className }: SettingsNavHeaderProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className={cn("px-3.5", className)}>
      <Typography.Text className="text-[11px] font-medium text-[var(--oh-text-dim)] uppercase tracking-wide leading-5">
        {t(text)}
      </Typography.Text>
    </div>
  );
}
