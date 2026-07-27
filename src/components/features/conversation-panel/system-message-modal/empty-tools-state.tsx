import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

export function EmptyToolsState() {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex items-center justify-center h-full p-4">
      <Typography.Text className="text-[var(--oh-muted)]">
        {t(I18nKey.SYSTEM_MESSAGE_MODAL$NO_TOOLS)}
      </Typography.Text>
    </div>
  );
}
