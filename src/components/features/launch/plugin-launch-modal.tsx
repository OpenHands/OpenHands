import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { hasPluginParameters } from "#/utils/plugin-parameters";
import { PluginLaunchPluginSection } from "./plugin-launch-plugin-section";
import {
  SchemaFormField,
  applySchemaDefaults,
} from "#/utils/plugin-launch-preflight";

interface PluginLaunchModalProps {
  plugins: PluginSpec[];
  parametersSupported: boolean;
  message?: string;
  isLoading?: boolean;
  /** Submit target: start a conversation (default) or install an automation. */
  mode?: "conversation" | "automation";
  /** Preflight response for the (single) plugin, when the backend serves one. */
  preflight?: {
    description?: string | null;
    resolvedRef?: string | null;
    fields: SchemaFormField[];
    suggestedTrigger?: {
      type: string;
      schedule?: string;
      timezone?: string;
    } | null;
  } | null;
  onStartConversation: (plugins: PluginSpec[], initialMessage?: string) => void;
  /** Required when mode === "automation". */
  onInstallAutomation?: (
    plugins: PluginSpec[],
    trigger?: { type: "cron"; schedule: string; timezone: string } | null,
  ) => void;
  onClose: () => void;
}

interface ExpandedState {
  [key: number]: boolean;
}

export function PluginLaunchModal({
  plugins,
  parametersSupported,
  message,
  isLoading = false,
  mode = "conversation",
  preflight = null,
  onStartConversation,
  onInstallAutomation,
  onClose,
}: PluginLaunchModalProps) {
  const { t } = useTranslation("openhands");
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

  // Manifest-aware automation install (OpenHands#18793): when the backend
  // serves a plugin-spec preflight, seed typed fields from the schema and
  // offer an editable trigger defaulted from the manifest suggestion.
  const isAutomationInstall = mode === "automation";
  const preflightFields = isAutomationInstall ? (preflight?.fields ?? []) : [];

  // Seed defaults once when the preflight arrives; later user edits win.
  React.useEffect(() => {
    if (isAutomationInstall && preflight && preflight.fields.length > 0) {
      setPluginConfigs((prev) =>
        prev.map((plugin) => ({
          ...plugin,
          parameters: applySchemaDefaults(
            plugin.parameters ?? {},
            preflight.fields,
          ),
        })),
      );
      // The preflight arrives after mount, so the initial expand state
      // (computed from empty parameters) leaves these sections collapsed.
      // Expand every section that now has schema-seeded parameters.
      setExpandedSections((prev) => {
        const next = { ...prev };
        pluginConfigs.forEach((_, index) => {
          next[index] = true;
        });
        return next;
      });
    }
  }, [isAutomationInstall, preflight]);

  const suggestedTrigger = isAutomationInstall
    ? (preflight?.suggestedTrigger ?? null)
    : null;
  const isCronSuggestion = suggestedTrigger?.type === "cron";
  const [triggerSchedule, setTriggerSchedule] = React.useState("");
  const [triggerTimezone, setTriggerTimezone] = React.useState("UTC");

  // The preflight arrives asynchronously; sync the editable inputs once when
  // its cron suggestion lands. Later user edits win over the suggestion.
  React.useEffect(() => {
    if (isCronSuggestion && suggestedTrigger) {
      setTriggerSchedule(suggestedTrigger.schedule ?? "");
      setTriggerTimezone(suggestedTrigger.timezone ?? "UTC");
    }
  }, [preflight]);

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

  const unsupportedParameters =
    !parametersSupported &&
    // Automation installs map `parameters` to the plugin's `config`, which the
    // automation service validates itself — no agent-server dependency.
    mode !== "automation" &&
    hasPluginParameters(pluginConfigs);

  const handleStartConversation = () => {
    if (unsupportedParameters) return;
    if (mode === "automation") {
      if (isCronSuggestion) {
        onInstallAutomation?.(pluginConfigs, {
          type: "cron",
          schedule: triggerSchedule,
          timezone: triggerTimezone,
        });
        return;
      }
      onInstallAutomation?.(pluginConfigs);
      return;
    }
    onStartConversation(pluginConfigs, message);
  };

  // The generic title already includes the verb ("Launch Plugin"), so only a
  // single plugin's display name is prefixed with "Launch".
  const modalTitle =
    pluginConfigs.length === 1
      ? `${t(I18nKey.LAUNCH$MODAL_TITLE)} ${getPluginDisplayName(pluginConfigs[0])}`
      : t(I18nKey.LAUNCH$MODAL_TITLE_GENERIC);

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="plugin-launch-modal"
        className={cn(
          "relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-border max-h-[80vh]",
          modalWidthClassName("md"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <ModalCloseButton onClose={onClose} testId="close-button" />
        <Typography.H2 className="pr-6">{modalTitle}</Typography.H2>

        {message && <p className="text-sm text-contrast">{message}</p>}

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
                  schemaFields={preflightFields}
                  onParameterChange={updateParameter}
                />
              ))}
            </div>
          )}

          {pluginsWithoutParams.length > 0 && (
            <div className={cn(pluginsWithParams.length > 0 && "mt-4")}>
              <Typography.H3 className="mb-2 text-contrast">
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
                    <div className="text-xs text-contrast mt-1">
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

        {unsupportedParameters && (
          <p
            role="alert"
            data-testid="plugin-parameters-unsupported"
            className="text-sm text-feedback-error"
          >
            {t(I18nKey.LAUNCH$PARAMETERS_UNSUPPORTED_LOCAL)}
          </p>
        )}

        {isAutomationInstall && preflight?.description && (
          <p
            data-testid="plugin-preflight-description"
            className="text-sm text-contrast"
          >
            {preflight.description}
          </p>
        )}

        {isAutomationInstall && isCronSuggestion && (
          <div
            data-testid="plugin-trigger-section"
            className="space-y-2 rounded-md bg-tertiary p-3"
          >
            <label
              htmlFor="plugin-trigger-schedule"
              className="text-sm font-medium text-contrast"
            >
              {t(I18nKey.LAUNCH$TRIGGER_SCHEDULE_LABEL)}
            </label>
            <input
              id="plugin-trigger-schedule"
              data-testid="plugin-trigger-schedule"
              type="text"
              value={triggerSchedule}
              onChange={(e) => setTriggerSchedule(e.target.value)}
              placeholder={t(I18nKey.LAUNCH$TRIGGER_SCHEDULE_PLACEHOLDER)}
              className="w-full rounded-md border border-border bg-basic px-3 py-2 text-sm"
            />
            <label
              htmlFor="plugin-trigger-timezone"
              className="text-sm font-medium text-contrast"
            >
              {t(I18nKey.LAUNCH$TRIGGER_TIMEZONE_LABEL)}
            </label>
            <input
              id="plugin-trigger-timezone"
              data-testid="plugin-trigger-timezone"
              type="text"
              value={triggerTimezone}
              onChange={(e) => setTriggerTimezone(e.target.value)}
              placeholder={t(I18nKey.LAUNCH$TRIGGER_TIMEZONE_PLACEHOLDER)}
              className="w-full rounded-md border border-border bg-basic px-3 py-2 text-sm"
            />
            <p className="text-xs text-contrast">
              {t(I18nKey.LAUNCH$TRIGGER_EDIT_HINT, {
                schedule: suggestedTrigger?.schedule ?? "",
                interpolation: { escapeValue: false },
              })}
            </p>
          </div>
        )}

        <div className="pt-4 border-t border-border-subtle">
          <div className="flex items-start gap-3 mb-4">
            <input
              id="trust-checkbox"
              data-testid="trust-checkbox"
              type="checkbox"
              checked={trustConfirmed}
              onChange={(e) => setTrustConfirmed(e.target.checked)}
              className="mt-1 h-4 w-4 flex-shrink-0"
            />
            <label htmlFor="trust-checkbox" className="text-sm text-contrast">
              {t(I18nKey.LAUNCH$TRUST_PLUGIN_CHECKBOX, {
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
              isDisabled={isLoading || !trustConfirmed || unsupportedParameters}
              className="px-4"
            >
              {isLoading
                ? t(I18nKey.LAUNCH$STARTING)
                : mode === "automation"
                  ? t(I18nKey.LAUNCH$INSTALL_AUTOMATION)
                  : t(I18nKey.LAUNCH$START_CONVERSATION)}
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
