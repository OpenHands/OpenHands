import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

export function HomeHeaderTitle() {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex min-h-[80px] w-full items-center justify-center py-2">
      <Typography.H1 className="w-full text-center leading-normal">
        {t(I18nKey.HOME$LETS_START_BUILDING)}
      </Typography.H1>
    </div>
  );
}
