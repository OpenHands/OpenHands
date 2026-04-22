import { useTranslation } from "react-i18next";
import { IoChevronDown, IoChevronForward } from "react-icons/io5";
import { I18nKey } from "#/i18n/declaration";
import { PluginSpec } from "#/api/conversation-service/v1-conversation-service.types";
import { Typography } from "#/ui/typography";
import { PluginLaunchParameterInput } from "./plugin-launch-parameter-input";

export interface PluginLaunchPluginSectionProps {
  plugin: PluginSpec;
  originalIndex: number;
  isExpanded: boolean;
  onToggle: () => void;
  getPluginDisplayName: (plugin: PluginSpec) => string;
  onParameterChange: (
    pluginIndex: number,
    paramKey: string,
    value: unknown,
  ) => void;
}

export function PluginLaunchPluginSection({
  plugin,
  originalIndex,
  isExpanded,
  onToggle,
  getPluginDisplayName,
  onParameterChange,
}: PluginLaunchPluginSectionProps) {
  const { t } = useTranslation();
  const hasParams =
    plugin.parameters && Object.keys(plugin.parameters).length > 0;

  if (!hasParams) {
    return null;
  }

  return (
    <div className="rounded-lg border border-tertiary bg-tertiary">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-base-tertiary rounded-t-lg cursor-pointer"
        data-testid={`plugin-section-${originalIndex}`}
      >
        <Typography.Text className="text-base font-medium">
          {getPluginDisplayName(plugin)}
        </Typography.Text>
        {isExpanded ? (
          <IoChevronDown className="h-5 w-5 text-white" />
        ) : (
          <IoChevronForward className="h-5 w-5 text-white" />
        )}
      </button>

      {isExpanded && (
        <div className="border-t border-tertiary px-4 pb-3">
          {plugin.ref && (
            <div className="mb-2 text-sm text-white">
              {t(I18nKey.LAUNCH$PLUGIN_REF)} {plugin.ref}
            </div>
          )}
          {plugin.repo_path && (
            <div className="mb-2 text-sm text-white">
              {t(I18nKey.LAUNCH$PLUGIN_PATH)} {plugin.repo_path}
            </div>
          )}
          <div className="flex flex-col gap-4">
            {Object.entries(plugin.parameters || {}).map(([key, value]) => (
              <PluginLaunchParameterInput
                key={key}
                pluginIndex={originalIndex}
                paramKey={key}
                paramValue={value}
                onParameterChange={onParameterChange}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
