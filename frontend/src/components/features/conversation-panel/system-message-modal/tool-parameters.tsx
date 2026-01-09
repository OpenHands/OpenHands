import { useTranslation } from "react-i18next";
import ReactJsonView from "@microlink/react-json-view";
import { JSON_VIEW_THEME } from "#/utils/constants";
import { Typography } from "#/ui/typography";

interface ToolParametersProps {
  parameters: Record<string, unknown>;
}

export function ToolParameters({ parameters }: ToolParametersProps) {
  const { t } = useTranslation();

  return (
    <div className="mt-2" data-testid="tool-parameters">
      <Typography.Text className="text-sm font-semibold text-gray-300">
        {t("SYSTEM_MESSAGE_MODAL$PARAMETERS")}
      </Typography.Text>
      <div className="text-sm mt-2 p-3 bg-gray-900 rounded-md overflow-auto text-gray-300 max-h-[400px] shadow-inner">
        <ReactJsonView name={false} src={parameters} theme={JSON_VIEW_THEME} />
      </div>
    </div>
  );
}
