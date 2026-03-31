import React from "react";
import { useTranslation } from "react-i18next";
import { IoClose } from "react-icons/io5";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { PluginSpec } from "#/api/conversation-service/v1-conversation-service.types";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { PluginLaunchPluginSection } from "./plugin-launch-plugin-section";

interface PluginLaunchModalProps {
  plugins: PluginSpec[];
  message?: string;
  isLoading?: boolean;
  onStartConversation: (plugins: PluginSpec[], initialMessage?: string) => void;
  onClose: () => void;
}

interface ExpandedState {
  [key: number]: boolean;
}

export function PluginLaunchModal({
  plugins,
  message,
  isLoading = false,
  onStartConversation,
  onClose,
}: PluginLaunchModalProps) {
  const { t } = useTranslation();
  const [pluginConfigs, setPluginConfigs] =
    React.useState<PluginSpec[]>(plugins);
  const [expandedSections, setExpandedSections] = React.useState<ExpandedState>(
    () => {
      // Initially expand plugins that have parameters
      const initial: ExpandedState = {};
      plugins.forEach((plugin, index) => {
        if (plugin.parameters && Object.keys(plugin.parameters).length > 0) {
          initial[index] = true;
        }
      });
      return initial;
    },
  );
  const [trustConfirmed, setTrustConfirmed] = React.useState(false);

  const pluginsWithParams = pluginConfigs.filter(
    (p) => p.parameters && Object.keys(p.parameters).length > 0,
  );
  const pluginsWithoutParams = pluginConfigs.filter(
    (p) => !p.parameters || Object.keys(p.parameters).length === 0,
  );

  const toggleSection = (index: number) => {
    setExpandedSections((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const updateParameter = (
    pluginIndex: number,
    paramKey: string,
    value: unknown,
  ) => {
    setPluginConfigs((prev) => {
      const updated = [...prev];
      const plugin = { ...updated[pluginIndex] };
      plugin.parameters = {
        ...plugin.parameters,
        [paramKey]: value,
      };
      updated[pluginIndex] = plugin;
      return updated;
    });
  };

  const getPluginDisplayName = (plugin: PluginSpec): string => {
    const { source, repo_path: repoPath } = plugin;

    // If repo_path is specified, show the plugin name from the path
    if (repoPath) {
      const pathParts = repoPath.split("/");
      const pluginName = pathParts[pathParts.length - 1];
      return pluginName;
    }

    // Otherwise show the repo name
    if (source.startsWith("github:")) {
      return source.replace("github:", "");
    }
    if (source.includes("/")) {
      const parts = source.split("/");
      return parts[parts.length - 1].replace(".git", "");
    }
    return source;
  };

  const getPluginSourceInfo = (plugin: PluginSpec): string => {
    const { source } = plugin;
    if (source.startsWith("github:")) {
      return source.replace("github:", "");
    }
    if (source.includes("github.com/")) {
      return source.split("github.com/")[1]?.replace(".git", "") || source;
    }
    return source;
  };

  const getUniqueSources = (): string[] => {
    const sources = pluginConfigs.map((plugin) => getPluginSourceInfo(plugin));
    return [...new Set(sources)];
  };

  const handleStartConversation = () => {
    onStartConversation(pluginConfigs, message);
  };

  const modalTitle =
    pluginConfigs.length === 1
      ? getPluginDisplayName(pluginConfigs[0])
      : t(I18nKey.LAUNCH$MODAL_TITLE_GENERIC);

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="plugin-launch-modal"
        className="bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-tertiary w-[500px] max-w-[90vw] max-h-[80vh]"
      >
        <div className="flex w-full items-center justify-between">
          <Typography.H2>
            {t(I18nKey.LAUNCH$MODAL_TITLE)} {modalTitle}
          </Typography.H2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-white hover:bg-tertiary cursor-pointer"
            aria-label="Close"
            data-testid="close-button"
          >
            <IoClose className="h-5 w-5" />
          </button>
        </div>

        {message && <p className="text-sm text-white">{message}</p>}

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {pluginsWithParams.length > 0 && (
            <div className="space-y-3">
              {pluginConfigs.map((plugin, index) => (
                <PluginLaunchPluginSection
                  key={`plugin-${index}`}
                  plugin={plugin}
                  originalIndex={index}
                  isExpanded={!!expandedSections[index]}
                  onToggle={() => toggleSection(index)}
                  getPluginDisplayName={getPluginDisplayName}
                  onParameterChange={updateParameter}
                />
              ))}
            </div>
          )}

          {pluginsWithoutParams.length > 0 && (
            <div className={cn(pluginsWithParams.length > 0 && "mt-4")}>
              <Typography.H3 className="mb-2 text-white">
                {pluginsWithParams.length > 0
                  ? t(I18nKey.LAUNCH$ADDITIONAL_PLUGINS)
                  : t(I18nKey.LAUNCH$PLUGINS)}
              </Typography.H3>
              <div className="space-y-2">
                {pluginsWithoutParams.map((plugin, index) => (
                  <div
                    key={`simple-plugin-${index}`}
                    className="rounded-md bg-tertiary px-3 py-2 text-sm"
                  >
                    <div className="font-medium">
                      {getPluginDisplayName(plugin)}
                    </div>
                    <div className="text-xs text-white mt-1">
                      {getPluginSourceInfo(plugin)}
                      {plugin.repo_path && (
                        <span className="ml-1">/ {plugin.repo_path}</span>
                      )}
                      {plugin.ref && (
                        <span className="ml-2">@ {plugin.ref}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="pt-4 border-t border-tertiary">
          <div className="flex items-start gap-3 mb-4">
            <input
              id="trust-checkbox"
              data-testid="trust-checkbox"
              type="checkbox"
              checked={trustConfirmed}
              onChange={(e) => setTrustConfirmed(e.target.checked)}
              className="mt-1 h-4 w-4 flex-shrink-0"
            />
            <label htmlFor="trust-checkbox" className="text-sm text-white">
              {t(I18nKey.LAUNCH$TRUST_SKILL_CHECKBOX, {
                sources: getUniqueSources().join(", "),
                interpolation: { escapeValue: false },
              })}
            </label>
          </div>
          <div className="flex w-full justify-end mt-8">
            <BrandButton
              testId="start-conversation-button"
              type="button"
              variant="primary"
              onClick={handleStartConversation}
              isDisabled={isLoading || !trustConfirmed}
              className="px-4"
            >
              {isLoading
                ? t(I18nKey.LAUNCH$STARTING)
                : t(I18nKey.LAUNCH$START_CONVERSATION)}
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
