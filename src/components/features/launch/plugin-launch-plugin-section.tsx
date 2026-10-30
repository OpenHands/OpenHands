import { useTranslation } from "react-i18next";
import { useEffect, useRef } from "react";
import { IoChevronDown, IoChevronForward } from "react-icons/io5";
import { I18nKey } from "#/i18n/declaration";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { Typography } from "#/ui/typography";
import { PluginLaunchParameterInput } from "./plugin-launch-parameter-input";
import type { SchemaFormField } from "#/utils/plugin-launch-preflight";

export interface PluginLaunchPluginSectionProps {
  plugin: PluginSpec;
  originalIndex: number;
  isExpanded: boolean;
  onToggle: () => void;
  getPluginDisplayName: (plugin: PluginSpec) => string;
  /** Manifest schema fields (automation mode with preflight), if available. */
  schemaFields?: SchemaFormField[];
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
  schemaFields,
  onParameterChange,
}: PluginLaunchPluginSectionProps) {
  const { t } = useTranslation("openhands");
  const hasParams =
    plugin.parameters && Object.keys(plugin.parameters).length > 0;
  const expandedRef = useRef<HTMLDivElement>(null);

  // Scroll the expanded fields into view so the user sees them without
  // hunting. Fires when the section auto-expands from the preflight arrival
  // or when the user toggles it open.
  useEffect(() => {
    if (isExpanded && expandedRef.current) {
      expandedRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [isExpanded]);

  if (!hasParams) {
    return null;
  }

  return (
    <div className="rounded-lg border border-border bg-tertiary">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-interactive-hover rounded-t-lg cursor-pointer"
        data-testid={`plugin-section-${originalIndex}`}
      >
        <Typography.Text className="text-base font-normal">
          {getPluginDisplayName(plugin)}
        </Typography.Text>
        {isExpanded ? (
          <IoChevronDown className="h-5 w-5 text-contrast" />
        ) : (
          <IoChevronForward className="h-5 w-5 text-contrast" />
        )}
      </button>

      {isExpanded && (
        <div
          ref={expandedRef}
          className="border-t border-border-subtle px-4 pb-4 pt-3"
        >
          {plugin.ref && (
            <div className="mb-2 text-sm text-contrast">
              {t(I18nKey.LAUNCH$PLUGIN_REF)} {plugin.ref}
            </div>
          )}
          {plugin.repo_path && (
            <div className="mb-2 text-sm text-contrast">
              {t(I18nKey.LAUNCH$PLUGIN_PATH)} {plugin.repo_path}
            </div>
          )}
          <div className="flex flex-col gap-4">
            {Object.entries(plugin.parameters || {}).map(([key, value]) => {
              const field = schemaFields?.find((f) => f.key === key);
              return (
                <PluginLaunchParameterInput
                  key={key}
                  pluginIndex={originalIndex}
                  paramKey={key}
                  paramValue={value}
                  label={field?.label}
                  options={field?.options}
                  helpText={field?.helpText}
                  widget={field?.widget}
                  onParameterChange={onParameterChange}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
