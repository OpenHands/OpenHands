import React, { useEffect, useMemo, useRef, useState } from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useConfig } from "#/hooks/query/use-config";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { SettingsFieldSchema } from "#/types/settings";
import { Typography } from "#/ui/typography";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  resolveSchemaFieldDescription,
  resolveSchemaFieldLabel,
} from "#/utils/sdk-settings-field-metadata";
import { formatCommand, tokenizeCommand } from "#/utils/shell-tokenize";
import type { ACPProviderConfig } from "#/api/option-service/option.types";

const ENABLE_SUB_AGENTS_FIELD_KEY = "enable_sub_agents";
const CUSTOM_PRESET = "custom";
const CUSTOM_MODEL_KEY = "__custom__";
const EMPTY_ACP_PROVIDERS: ACPProviderConfig[] = [];

function findEnableSubAgentsField(
  fields: SettingsFieldSchema[] | undefined,
): SettingsFieldSchema | undefined {
  return fields?.find((field) => field.key === ENABLE_SUB_AGENTS_FIELD_KEY);
}

function getEnableSubAgentsValue(
  settingsValue: unknown,
  field: SettingsFieldSchema | undefined,
) {
  if (typeof settingsValue === "boolean") return settingsValue;
  return field?.default === true;
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((v): v is string => typeof v === "string")
    : [];
}

function detectPreset(
  commandText: string,
  providers: ACPProviderConfig[],
): string {
  const normalized = tokenizeCommand(commandText).join(" ");
  for (const provider of providers) {
    if (normalized === provider.default_command.join(" ")) {
      return provider.key;
    }
  }
  return CUSTOM_PRESET;
}

function isKnownModel(
  provider: ACPProviderConfig | undefined,
  model: string,
): boolean {
  return (
    provider?.available_models?.some(({ id }) => id === model.trim()) ?? false
  );
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default function AgentSettingsScreen() {
  const { t } = useTranslation();
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading: isSettingsLoading } = useSettings();
  const { data: config, isLoading: isConfigLoading } = useConfig();
  const { data: schema, isLoading: isSchemaLoading } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );

  const isAcpEnabled = !!config?.feature_flags?.enable_acp;
  const acpProviders = config?.acp_providers ?? EMPTY_ACP_PROVIDERS;

  // ── Sub-agents (OpenHands mode) ──────────────────────────────────────────
  const fields = useMemo(
    () => schema?.sections.flatMap((section) => section.fields),
    [schema],
  );
  const subAgentsField = findEnableSubAgentsField(fields);
  const initialSubAgentsEnabled = useMemo(
    () =>
      getEnableSubAgentsValue(
        settings?.agent_settings?.[ENABLE_SUB_AGENTS_FIELD_KEY],
        subAgentsField,
      ),
    [subAgentsField, settings?.agent_settings],
  );
  const [isSubAgentsEnabled, setIsSubAgentsEnabled] = useState(
    initialSubAgentsEnabled,
  );
  useEffect(() => {
    setIsSubAgentsEnabled(initialSubAgentsEnabled);
  }, [initialSubAgentsEnabled]);

  // ── ACP (ACP mode) ───────────────────────────────────────────────────────
  const [agentType, setAgentType] = useState<"openhands" | "acp">("openhands");
  const [commandText, setCommandText] = useState("");
  const [acpModel, setAcpModel] = useState("");
  const [isDirty, setIsDirty] = useState(false);

  const lastInitializedSettingsRef = useRef<unknown>(null);

  useEffect(() => {
    if (!settings || isConfigLoading) return;
    if (lastInitializedSettingsRef.current === settings) return;

    lastInitializedSettingsRef.current = settings;
    const kind = settings.agent_settings?.agent_kind;

    if (kind === "acp") {
      setAgentType("acp");
      const tokens = [
        ...toStringArray(settings.agent_settings?.acp_command),
        ...toStringArray(settings.agent_settings?.acp_args),
      ];
      const rawAcpServer = settings.agent_settings?.acp_server;
      const acpServer =
        typeof rawAcpServer === "string" ? rawAcpServer : undefined;
      const provider = acpProviders.find(({ key }) => key === acpServer);
      const joined = tokens.join(" ");
      setCommandText(joined || formatCommand(provider?.default_command ?? []));
      const savedModel = settings.agent_settings?.acp_model;
      const normalizedModel =
        typeof savedModel === "string" ? savedModel.trim() : "";
      setAcpModel(normalizedModel || provider?.default_model || "");
    } else {
      setAgentType("openhands");
      setCommandText("");
      setAcpModel("");
    }
    setIsDirty(false);
  }, [settings, acpProviders, isConfigLoading]);

  // ── Derived state ─────────────────────────────────────────────────────────
  const isAcp = agentType === "acp";
  const commandTokens = tokenizeCommand(commandText);
  const isAcpInvalid = isAcp && commandTokens.length === 0;
  const selectedPreset = detectPreset(commandText, acpProviders);
  const selectedProvider = acpProviders.find(
    ({ key }) => key === selectedPreset,
  );
  const isDefaultProviderCommand =
    !!selectedProvider &&
    commandTokens.join(" ") === selectedProvider.default_command.join(" ");
  const commandPlaceholder =
    formatCommand(acpProviders[0]?.default_command ?? []) ||
    "npx -y <package-name>";

  const modelSuggestions = selectedProvider?.available_models ?? [];
  const hasModelSuggestions = modelSuggestions.length > 0;
  const selectedModelIsSuggestion = isKnownModel(selectedProvider, acpModel);
  const selectedModelKey = selectedModelIsSuggestion
    ? acpModel
    : CUSTOM_MODEL_KEY;

  // ── Credential form ───────────────────────────────────────────────────────
  // Called unconditionally for hook-order stability; null arg → empty form.
  const credentialPreset =
    isAcp && selectedPreset !== CUSTOM_PRESET ? selectedPreset : null;
  const credentialForm = useAcpCredentialForm(
    credentialPreset,
    selectedProvider,
  );

  const subAgentsDirty = isSubAgentsEnabled !== initialSubAgentsEnabled;
  const settingsDirty = isDirty || (!isAcp && subAgentsDirty);
  const credentialsDirty = isAcp && credentialForm.isDirty;
  const canSave = settingsDirty || credentialsDirty;
  const isSavingAny = isPending || credentialForm.isSaving;

  // ── Save ──────────────────────────────────────────────────────────────────
  const handleSave = async () => {
    // Persist credentials first so they exist when the agent spec is applied.
    // silent=true when settings are also changing, so we show one "Saved" toast.
    if (isAcp && credentialForm.isDirty) {
      const ok = await credentialForm.save({ silent: settingsDirty });
      if (!ok) return;
      credentialForm.reset();
    }

    if (!settingsDirty) return;

    let agentSettingsDiff: Record<string, unknown>;

    if (isAcp) {
      agentSettingsDiff = {
        agent_kind: "acp",
        acp_server:
          selectedProvider && isDefaultProviderCommand
            ? selectedProvider.key
            : CUSTOM_PRESET,
        acp_command:
          selectedProvider && isDefaultProviderCommand ? [] : commandTokens,
        acp_model: acpModel.trim() || null,
      };
    } else if (isDirty) {
      // Agent-kind flip: backend resets to defaults, send kind alone.
      agentSettingsDiff = { agent_kind: "openhands" };
    } else {
      // Sub-agents toggle only.
      agentSettingsDiff = { enable_sub_agents: isSubAgentsEnabled };
    }

    saveSettings(
      { agent_settings_diff: agentSettingsDiff },
      {
        onError: (error) => {
          const message = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
        },
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
          setIsDirty(false);
        },
      },
    );
  };

  // ── Loading ───────────────────────────────────────────────────────────────
  if (isSettingsLoading || isSchemaLoading || isConfigLoading) {
    return <LlmSettingsInputsSkeleton />;
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div data-testid="agent-settings-screen" className="h-full relative">
      <div className="flex flex-col gap-8 pb-20">
        {/* Agent-type selector — only when ACP feature flag is on */}
        {isAcpEnabled && (
          <section className="grid gap-4 xl:grid-cols-2">
            <SettingsDropdownInput
              testId="agent-type-selector"
              name="agent-type"
              label={t(I18nKey.SETTINGS$AGENT)}
              items={[
                {
                  key: "openhands",
                  label: t(I18nKey.SETTINGS$AGENT_TYPE_OPENHANDS),
                },
                { key: "acp", label: t(I18nKey.SETTINGS$AGENT_TYPE_ACP) },
              ]}
              selectedKey={agentType}
              onSelectionChange={(key) => {
                if (!key) return;
                const newType = key as "openhands" | "acp";
                setAgentType(newType);
                if (newType === "acp" && !commandText) {
                  const preferred = acpProviders[0];
                  if (preferred) {
                    setCommandText(formatCommand(preferred.default_command));
                    setAcpModel(preferred.default_model || "");
                  }
                }
                setIsDirty(true);
              }}
            />
          </section>
        )}

        {/* OpenHands: sub-agents toggle */}
        {!isAcp && (
          <section className="grid gap-4 xl:grid-cols-2">
            {subAgentsField ? (
              <div className="flex flex-col gap-1.5">
                <SettingsSwitch
                  testId="agent-settings-enable-sub-agents"
                  isToggled={isSubAgentsEnabled}
                  onToggle={setIsSubAgentsEnabled}
                >
                  {resolveSchemaFieldLabel(
                    t,
                    subAgentsField.key,
                    subAgentsField.label,
                  )}
                </SettingsSwitch>
                {resolveSchemaFieldDescription(
                  t,
                  subAgentsField.key,
                  subAgentsField.description,
                ) ? (
                  <Typography.Paragraph className="text-tertiary-alt text-xs leading-5">
                    {resolveSchemaFieldDescription(
                      t,
                      subAgentsField.key,
                      subAgentsField.description,
                    )}
                  </Typography.Paragraph>
                ) : null}
              </div>
            ) : (
              <Typography.Paragraph className="text-tertiary-alt">
                {t(I18nKey.SETTINGS$SDK_SCHEMA_UNAVAILABLE)}
              </Typography.Paragraph>
            )}
          </section>
        )}

        {/* ACP: preset, command, model, credentials */}
        {isAcp && (
          <>
            <SettingsDropdownInput
              testId="agent-preset-selector"
              name="agent-preset"
              label={t(I18nKey.SETTINGS$AGENT_PRESET)}
              items={[
                ...acpProviders.map((provider) => ({
                  key: provider.key,
                  label: provider.display_name,
                })),
                {
                  key: CUSTOM_PRESET,
                  label: t(I18nKey.SETTINGS$AGENT_PRESET_CUSTOM),
                },
              ]}
              selectedKey={selectedPreset}
              onSelectionChange={(key) => {
                if (!key) return;
                const preset = String(key);
                const provider = acpProviders.find(
                  ({ key: k }) => k === preset,
                );
                if (provider) {
                  setCommandText(formatCommand(provider.default_command));
                  setAcpModel(provider.default_model || "");
                } else if (preset === CUSTOM_PRESET) {
                  setCommandText("");
                  setAcpModel("");
                }
                setIsDirty(true);
              }}
            />

            <div className="flex flex-col gap-2.5">
              <Typography.Text className="text-sm">
                {t(I18nKey.SETTINGS$MCP_COMMAND)}
              </Typography.Text>
              <textarea
                data-testid="agent-command-input"
                className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-white placeholder:italic placeholder:text-[#717888] min-h-[60px] resize-y focus:outline-none focus:border-white"
                value={commandText}
                placeholder={commandPlaceholder}
                onChange={(e) => {
                  const next = e.target.value;
                  // Sync model when provider changes via typed command.
                  const prevPreset = detectPreset(commandText, acpProviders);
                  const nextPreset = detectPreset(next, acpProviders);
                  if (nextPreset !== prevPreset) {
                    const nextProvider = acpProviders.find(
                      ({ key }) => key === nextPreset,
                    );
                    setAcpModel(nextProvider?.default_model || "");
                  }
                  setCommandText(next);
                  setIsDirty(true);
                }}
              />
              <Typography.Text className="text-xs text-[#717888]">
                {t(I18nKey.SETTINGS$AGENT_COMMAND_HINT)}
              </Typography.Text>
            </div>

            {/* Model: dropdown for known models, free text for custom */}
            <div className="flex flex-col gap-1.5">
              {hasModelSuggestions && (
                <SettingsDropdownInput
                  testId="agent-model-selector"
                  name="agent-model"
                  label={t(I18nKey.SETTINGS$AGENT_MODEL)}
                  items={[
                    ...modelSuggestions.map((m) => ({
                      key: m.id,
                      label: m.label,
                    })),
                    {
                      key: CUSTOM_MODEL_KEY,
                      label: t(I18nKey.SETTINGS$AGENT_PRESET_CUSTOM),
                    },
                  ]}
                  selectedKey={selectedModelKey}
                  onSelectionChange={(key) => {
                    if (!key) return;
                    const mk = String(key);
                    if (mk === CUSTOM_MODEL_KEY) {
                      setAcpModel("");
                    } else {
                      setAcpModel(mk);
                    }
                    setIsDirty(true);
                  }}
                />
              )}
              {selectedModelKey === CUSTOM_MODEL_KEY && (
                <SettingsInput
                  testId="agent-model-input"
                  label={
                    hasModelSuggestions
                      ? t(I18nKey.SETTINGS$AGENT_CUSTOM_MODEL)
                      : t(I18nKey.SETTINGS$AGENT_MODEL)
                  }
                  type="text"
                  className="w-full"
                  value={acpModel}
                  showOptionalTag
                  onChange={(value) => {
                    setAcpModel(value);
                    setIsDirty(true);
                  }}
                />
              )}
              <Typography.Text className="text-xs text-[#717888]">
                {t(I18nKey.SETTINGS$AGENT_MODEL_HINT)}
              </Typography.Text>
            </div>

            {/* Credentials section for built-in providers */}
            {credentialForm.fields.length > 0 && (
              <>
                <hr className="border-[#3D4046]" />
                <AcpCredentialsSection form={credentialForm} />
              </>
            )}
          </>
        )}
      </div>

      <div className="sticky bottom-0 bg-base py-4">
        <BrandButton
          testId="agent-save-button"
          type="button"
          variant="primary"
          isDisabled={isSavingAny || !canSave || isAcpInvalid}
          onClick={handleSave}
        >
          {isSavingAny
            ? t(I18nKey.SETTINGS$SAVING)
            : t(I18nKey.SETTINGS$SAVE_CHANGES)}
        </BrandButton>
      </div>
    </div>
  );
}
