import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

interface SettingsNavHeaderProps {
  text: I18nKey;
}

export function SettingsNavHeader({ text }: SettingsNavHeaderProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="px-3.5">
      <Typography.Text className="text-2xs font-medium text-text-dim uppercase tracking-wide leading-5">
        {t(text)}
      </Typography.Text>
    </div>
  );
}
