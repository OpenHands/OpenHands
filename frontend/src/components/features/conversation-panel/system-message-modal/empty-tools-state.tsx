import { useTranslation } from "react-i18next";
import { Typography } from "#/ui/typography";

export function EmptyToolsState() {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-center h-full p-4">
      <Typography.Text className="text-muted">
        {t("SYSTEM_MESSAGE_MODAL$NO_TOOLS")}
      </Typography.Text>
    </div>
  );
}
