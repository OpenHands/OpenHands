import { useTranslation } from "react-i18next";
import ReactJsonView from "@microlink/react-json-view";
import { JSON_VIEW_THEME } from "#/utils/constants";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

interface ToolParametersProps {
  parameters: Record<string, unknown>;
}

export function ToolParameters({ parameters }: ToolParametersProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="mt-2" data-testid="tool-parameters">
      <Typography.Text className="text-sm font-semibold text-text-tertiary">
        {t(I18nKey.SYSTEM_MESSAGE_MODAL$PARAMETERS)}
      </Typography.Text>
      <div className="text-sm mt-2 p-3 bg-base rounded-md overflow-auto text-text-tertiary max-h-100 border border-border">
        <ReactJsonView name={false} src={parameters} theme={JSON_VIEW_THEME} />
      </div>
    </div>
  );
}
