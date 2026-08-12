import React, { useCallback, useEffect, useRef, useState } from "react";
import { Navigate } from "react-router";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { useSettings } from "#/hooks/query/use-settings";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { formControlSwitchDescriptionClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { SettingsFieldSchema, SettingsValue } from "#/types/settings";
import {
  coerceFieldValue,
  normalizeFieldValue,
} from "#/utils/sdk-settings-schema";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  resolveSchemaFieldDescription,
  resolveSchemaFieldLabel,
} from "#/utils/sdk-settings-field-metadata";
import {
  ACP_PROVIDERS,
  ACP_CUSTOM_PRESET_KEY,
  buildAcpAgentSettingsDiff,
  getAcpPreferredDefaultModel,
  getAcpProvider,
  type ACPProviderConfig,
} from "#/constants/acp-providers";
import { parseCommand, formatCommand } from "#/utils/acp-command";

export const handle = { hideTitle: true };

type AgentType = "openhands" | "acp";

const ENABLE_SUB_AGENTS_FIELD_KEY = "enable_sub_agents";
const TOOL_CONCURRENCY_FIELD_KEY = "tool_concurrency_limit";
const COMMAND_PLACEHOLDER_FALLBACK = "npx -y <package-name>";
const ACP_CUSTOM_MODEL_KEY = "__custom_model__";

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((v): v is string => typeof v === "string")
    : [];
}

function detectPreset(
  commandText: string,
  providers: ACPProviderConfig[],
): string {
  const normalized = parseCommand(commandText).join(" ");
  for (const provider of providers) {
    if (normalized === provider.default_command.join(" ")) {
      return provider.key;
    }
  }
  return ACP_CUSTOM_PRESET_KEY;
}

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

function isKnownAcpModel(
  provider: ACPProviderConfig | undefined,
  model: string,
): boolean {
  return (
    provider?.available_models?.some(({ id }) => id === model.trim()) ?? false
  );
}

/**
 * Variant-specific AgentProfile fields derived from the form state. The
 * OpenHands branch omits `llm_profile_ref` — the profile editor supplies it.
 */
export type AgentProfileFieldsDraft =
  | {
      agent_kind: "openhands";
      enable_sub_agents: boolean;
      tool_concurrency_limit?: number;
    }
  | {
      agent_kind: "acp";
      acp_server: string;
      acp_model: string | null;
      acp_command: string | null;
      acp_args: string[] | null;
    };

/** Live form state the pure {@link buildAgentProfileFields} builder reads. */
export interface AgentProfileFieldsInput {
  isAcp: boolean;
  /** Detected ACP preset: a provider key or the ``custom`` sentinel. */
  selectedPreset: string;
  /** True when the command exactly matches the selected provider's default. */
  isDefaultProviderCommand: boolean;
  commandTokens: string[];
  acpModel: string;
  subAgentsEnabled: boolean;
  toolConcurrencyField?: SettingsFieldSchema;
  toolConcurrency: string | boolean;
}

/**
 * Translate the live Agent-settings form state into the variant-specific
 * AgentProfile fields. Pure (no React), so it can be unit-tested directly.
 *
 * ACP: a built-in provider on its default command stores **no** explicit
 * command (``acp_command: null`` — the profile resolver falls back to the
 * provider default); a customized or ``custom`` command is stored verbatim as a
 * shell string. OpenHands: reuses the schema-driven ``tool_concurrency_limit``
 * coercion, which **throws** on invalid input (callers catch at save time). A
 * blank concurrency field always emits an explicit value (the schema default
 * when the coercion is empty) rather than omitting the key — the profile
 * editor's save is a whole-profile overwrite (``mergeAgentProfileSaveInput``
 * spreads the stored profile under these fields), so omitting the key would
 * let a stale stored value silently survive an edit meant to clear it back to
 * the default (#1571 review). The backend field itself is a non-nullable
 * ``int`` with ``ge=1``, so the default — not ``null`` — is the value that
 * actually clears.
 */
export function buildAgentProfileFields(
  input: AgentProfileFieldsInput,
): AgentProfileFieldsDraft {
  const {
    isAcp,
    selectedPreset,
    isDefaultProviderCommand,
    commandTokens,
    acpModel,
    subAgentsEnabled,
    toolConcurrencyField,
    toolConcurrency,
  } = input;
  if (isAcp) {
    const isBuiltinDefault =
      isDefaultProviderCommand && selectedPreset !== ACP_CUSTOM_PRESET_KEY;
    return {
      agent_kind: "acp",
      acp_server: selectedPreset,
      acp_model: acpModel.trim() || null,
      acp_command: isBuiltinDefault
        ? null
        : formatCommand(commandTokens) || null,
      acp_args: null,
    };
  }
  const fields: Extract<AgentProfileFieldsDraft, { agent_kind: "openhands" }> =
    {
      agent_kind: "openhands",
      enable_sub_agents: subAgentsEnabled,
    };
  if (toolConcurrencyField) {
    // Reuse the schema-driven coercion/validation; throws on bad input.
    const coerced = coerceFieldValue(toolConcurrencyField, toolConcurrency);
    // A blank field coerces to `null`. Always emit an explicit value — never
    // omit the key — so a deliberate clear on an edit-save actually resets the
    // stored profile to the schema default, instead of the whole-profile merge
    // silently carrying the old value forward.
    const fallback =
      typeof toolConcurrencyField.default === "number"
        ? toolConcurrencyField.default
        : 1;
    fields.tool_concurrency_limit =
      coerced != null ? Number(coerced) : fallback;
  }
  return fields;
}

/**
 * Handle the embedded form exposes to its parent (the Agent-profile editor) so
 * it can read the current state and persist it as an AgentProfile.
 */
export interface AgentSettingsSaveControl {
  agentType: AgentType;
  /** False when the current form can't be saved (e.g. an empty ACP command). */
  isValid: boolean;
  /**
   * Build the variant-specific AgentProfile fields from the live form state.
   * Throws a user-facing Error on invalid input (e.g. a bad concurrency value).
   */
  buildAgentProfileFields: () => AgentProfileFieldsDraft;
  /** Shared ACP credential form (writes to global secrets). */
  credentials: {
    isDirty: boolean;
    save: (opts?: { silent?: boolean }) => Promise<boolean>;
    reset: () => void;
  };
}

interface AgentSettingsScreenProps {
  /**
   * Embedded mode reuses this form as the Agent-profile editor: it hides the
   * page header + the global Save button, seeds from `agentSettingsOverride`
   * instead of the live global settings, and reports its state through
   * `onSaveControlChange` so the parent can persist it as an AgentProfile.
   */
  embedded?: boolean;
  /**
   * When set (embedded mode), seed the form from this `agent_settings`-shaped
   * object instead of the live global settings — lets the editor open on a
   * stored profile's fields.
   */
  agentSettingsOverride?: Record<string, SettingsValue> | null;
  onSaveControlChange?: (control: AgentSettingsSaveControl) => void;
}

export function AgentSettingsScreen({
  embedded = false,
  agentSettingsOverride = null,
  onSaveControlChange,
}: AgentSettingsScreenProps = {}) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading } = useSettings();
  // In embedded (profile-editor) mode the parent seeds the form from a stored
  // profile via `agentSettingsOverride`; otherwise use the live global settings.
  const agentSettingsSource: Record<string, SettingsValue> | null =
    agentSettingsOverride ?? settings?.agent_settings ?? null;
  const { mutate: saveSettings, isPending: isSaving } = useSaveSettings();
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );

  // --- Sub-agents (OpenHands path) ---
  const fields = React.useMemo(
    () => schema?.sections.flatMap((section) => section.fields),
    [schema],
  );
  const subAgentsField = findEnableSubAgentsField(fields);
  const initialSubAgentsEnabled = React.useMemo(
    () =>
      getEnableSubAgentsValue(
        agentSettingsSource?.[ENABLE_SUB_AGENTS_FIELD_KEY],
        subAgentsField,
      ),
    [subAgentsField, agentSettingsSource],
  );
  const [subAgentsEnabled, setSubAgentsEnabled] = useState(
    initialSubAgentsEnabled,
  );

  // --- Parallel tool calls (OpenHands path) ---
  // Surfaced only when the backend schema exposes the field, so older
  // agent-servers that predate ``tool_concurrency_limit`` hide it cleanly.
  const toolConcurrencyField = fields?.find(
    (field) => field.key === TOOL_CONCURRENCY_FIELD_KEY,
  );
  const initialToolConcurrency = React.useMemo(() => {
    if (!toolConcurrencyField) return "";
    const raw = agentSettingsSource?.[TOOL_CONCURRENCY_FIELD_KEY];
    return normalizeFieldValue(toolConcurrencyField, raw);
  }, [toolConcurrencyField, agentSettingsSource]);
  const [toolConcurrency, setToolConcurrency] = useState<string | boolean>(
    initialToolConcurrency,
  );

  // --- ACP path ---
  const [agentType, setAgentType] = useState<AgentType>("openhands");
  const [commandText, setCommandText] = useState("");
  const [acpModel, setAcpModel] = useState("");
  const [isCustomAcpModel, setIsCustomAcpModel] = useState(false);
  const [isDirty, setIsDirty] = useState(false);

  // ACP credentials live alongside the agent spec, so the page owns the
  // credential form and a single Save persists both. Called unconditionally
  // (no-ops to empty fields for a non-ACP / custom command) to keep hook order
  // stable across the ``isLoading`` early-return below; ``detectPreset`` is a
  // cheap pure lookup.
  const acpPresetForCreds =
    agentType === "acp" ? detectPreset(commandText, ACP_PROVIDERS) : null;
  const acpCredentialForm = useAcpCredentialForm(
    acpPresetForCreds && acpPresetForCreds !== ACP_CUSTOM_PRESET_KEY
      ? acpPresetForCreds
      : null,
  );

  const lastInitializedSettingsRef = useRef<unknown>(null);
  const loadedAcpServerRef = useRef<string | null>(null);
  const loadedCommandTextRef = useRef<string>("");

  useEffect(() => {
    // Seed from the profile override (embedded) or the live global settings.
    const source = agentSettingsOverride ?? settings?.agent_settings ?? null;
    if (!source && !settings) return;
    const initIdentity = agentSettingsOverride ?? settings;
    if (lastInitializedSettingsRef.current === initIdentity) return;

    lastInitializedSettingsRef.current = initIdentity;
    const kind = source?.agent_kind;

    if (kind === "acp") {
      setAgentType("acp");

      const rawAcpServer = source?.acp_server;
      const acpServer =
        typeof rawAcpServer === "string" ? rawAcpServer : undefined;
      const provider = getAcpProvider(acpServer);
      const storedCommand = toStringArray(source?.acp_command);
      const effectiveBaseCommand =
        storedCommand.length > 0
          ? storedCommand
          : (provider?.default_command ?? []);
      const tokens = [
        ...effectiveBaseCommand,
        ...toStringArray(source?.acp_args),
      ];
      const renderedCommandText =
        tokens.length > 0 ? formatCommand(tokens) : "";
      setCommandText(renderedCommandText);
      loadedAcpServerRef.current = acpServer ?? null;
      loadedCommandTextRef.current = renderedCommandText;

      const savedModel = source?.acp_model;
      const normalizedSavedModel =
        typeof savedModel === "string" ? savedModel.trim() : "";
      setAcpModel(
        normalizedSavedModel || getAcpPreferredDefaultModel(acpServer) || "",
      );
      setIsCustomAcpModel(
        !!normalizedSavedModel &&
          (!provider || !isKnownAcpModel(provider, normalizedSavedModel)),
      );
    } else {
      setAgentType("openhands");
      setCommandText("");
      setAcpModel("");
      loadedAcpServerRef.current = null;
      loadedCommandTextRef.current = "";
      setIsCustomAcpModel(false);
    }
    setIsDirty(false);
  }, [settings, agentSettingsOverride]);

  // Sync the sub-agents toggle when settings reload
  useEffect(() => {
    setSubAgentsEnabled(initialSubAgentsEnabled);
  }, [initialSubAgentsEnabled]);

  // Sync the parallel-tool-calls input when settings reload
  useEffect(() => {
    setToolConcurrency(initialToolConcurrency);
  }, [initialToolConcurrency]);

  // --- Embedded (Agent-profile editor) save control ---
  // Ref-backed so the exposed builder/credential fns read the freshest state at
  // call time without re-emitting the control on every keystroke (mirrors
  // ``sdk-section-page``). The body is (re)assigned during render below.
  const buildFieldsRef = useRef<() => AgentProfileFieldsDraft>(() => ({
    agent_kind: "openhands",
    enable_sub_agents: false,
  }));
  const stableBuildFields = useCallback(() => buildFieldsRef.current(), []);
  const credFormRef = useRef(acpCredentialForm);
  credFormRef.current = acpCredentialForm;
  const stableCredSave = useCallback(
    (opts?: { silent?: boolean }) => credFormRef.current.save(opts),
    [],
  );
  const stableCredReset = useCallback(() => credFormRef.current.reset(), []);

  // Validity/kind are computed here (before the loading early-return) so the
  // emit effect can depend on them; the full ACP derivation lives after it.
  const acpCommandEmpty =
    agentType === "acp" && parseCommand(commandText).length === 0;
  const embeddedCredentialsDirty = acpCredentialForm.isDirty;
  useEffect(() => {
    if (!embedded || !onSaveControlChange) return;
    onSaveControlChange({
      agentType,
      isValid: !acpCommandEmpty,
      buildAgentProfileFields: stableBuildFields,
      credentials: {
        isDirty: embeddedCredentialsDirty,
        save: stableCredSave,
        reset: stableCredReset,
      },
    });
  }, [
    embedded,
    onSaveControlChange,
    agentType,
    acpCommandEmpty,
    embeddedCredentialsDirty,
    stableBuildFields,
    stableCredSave,
    stableCredReset,
  ]);

  if (isLoading) return null;

  const isAcp = agentType === "acp";
  const commandTokens = parseCommand(commandText);
  const isAcpInvalid = isAcp && commandTokens.length === 0;
  const selectedPreset = detectPreset(commandText, ACP_PROVIDERS);
  const selectedProvider = getAcpProvider(selectedPreset);
  const modelSuggestions = selectedProvider?.available_models ?? [];
  const hasModelSuggestions = modelSuggestions.length > 0;
  const selectedModelIsSuggestion = isKnownAcpModel(selectedProvider, acpModel);
  const selectedModelKey =
    isCustomAcpModel || !selectedModelIsSuggestion
      ? ACP_CUSTOM_MODEL_KEY
      : acpModel;
  const isDefaultProviderCommand =
    !!selectedProvider &&
    commandTokens.join(" ") === selectedProvider.default_command.join(" ");
  const commandPlaceholder =
    formatCommand(ACP_PROVIDERS[0]?.default_command ?? []) ||
    COMMAND_PLACEHOLDER_FALLBACK;

  // Assign the embedded control's field builder from the live render state.
  // The mapping itself lives in the pure `buildAgentProfileFields` (unit-
  // tested); this closure just snapshots the current state. Throws only when
  // called (at save time), never during render.
  buildFieldsRef.current = (): AgentProfileFieldsDraft =>
    buildAgentProfileFields({
      isAcp,
      selectedPreset,
      isDefaultProviderCommand,
      commandTokens,
      acpModel,
      subAgentsEnabled,
      toolConcurrencyField,
      toolConcurrency,
    });

  // Dirty tracking: for OpenHands path, also check sub-agents toggle and the
  // parallel-tool-calls input.
  const isOpenHandsDirty =
    !isAcp &&
    (subAgentsEnabled !== initialSubAgentsEnabled ||
      toolConcurrency !== initialToolConcurrency);
  const settingsDirty = isDirty || isOpenHandsDirty;
  // The single Save covers both the agent spec and ACP credentials, so it is
  // active when either changed, and shows "Saving…" while either is in flight.
  // ``isDirty`` is already false off the ACP path (no credential fields), so no
  // ``isAcp`` guard is needed.
  const credentialsDirty = acpCredentialForm.isDirty;
  const isAnyDirty = settingsDirty || credentialsDirty;
  const isSavingAny = isSaving || acpCredentialForm.isSaving;

  const handleSave = async () => {
    // Persist ACP credentials first (if any were typed) so they exist when the
    // agent spec is applied. When the spec is also changing, save silently so
    // the settings save below owns the single "Saved" toast (otherwise the user
    // sees it twice); a credentials-only save shows its own toast. Errors always
    // toast and abort.
    if (acpCredentialForm.isDirty) {
      const ok = await acpCredentialForm.save({ silent: settingsDirty });
      if (!ok) return;
      acpCredentialForm.reset();
    }

    // Only write the agent spec when it actually changed — a credentials-only
    // edit must not re-push unchanged settings (or double-toast).
    if (!settingsDirty) return;

    if (isAcp) {
      const useDefault = !!(selectedProvider && isDefaultProviderCommand);
      const loadedServer = loadedAcpServerRef.current;
      const commandUnchanged = commandText === loadedCommandTextRef.current;
      const loadedServerIsUnknown =
        !!loadedServer &&
        loadedServer !== ACP_CUSTOM_PRESET_KEY &&
        !ACP_PROVIDERS.some((p) => p.key === loadedServer);
      const preserveUnknownServer =
        isAcp && commandUnchanged && loadedServerIsUnknown;
      const providerKey = preserveUnknownServer
        ? (loadedServer as string)
        : selectedProvider && isDefaultProviderCommand
          ? selectedProvider.key
          : ACP_CUSTOM_PRESET_KEY;
      // ``model: undefined`` lets buildAcpAgentSettingsDiff seed the
      // provider's preferred default for built-in keys; for the custom preset
      // it falls through to ``null`` since custom has no default.
      const agentSettingsDiff = buildAcpAgentSettingsDiff(providerKey, {
        command: useDefault ? [] : commandTokens,
        model: acpModel.trim() || undefined,
        allowUnknownServer: preserveUnknownServer,
      });

      if (!agentSettingsDiff) return;

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
    } else {
      // OpenHands path: save agent_kind + sub-agents toggle + parallel tool calls
      const agentSettingsDiff: Record<string, SettingsValue> = {
        agent_kind: "openhands",
        enable_sub_agents: subAgentsEnabled,
      };

      if (toolConcurrencyField) {
        let coerced: SettingsValue;
        try {
          // Reuse the schema-driven coercion + min/max validation rather than
          // re-implementing it; throws a user-facing message on bad input.
          coerced = coerceFieldValue(toolConcurrencyField, toolConcurrency);
        } catch (error) {
          displayErrorToast(
            error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
          );
          return;
        }
        // ``tool_concurrency_limit`` is a non-nullable int (default 1); skip an
        // empty input rather than sending ``null`` the backend would reject.
        if (coerced != null) {
          agentSettingsDiff[TOOL_CONCURRENCY_FIELD_KEY] = coerced;
        }
      }

      saveSettings(
        {
          agent_settings_diff: agentSettingsDiff,
        },
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
    }
  };

  // Sub-agents field metadata for OpenHands section
  const subAgentsLabel = subAgentsField
    ? resolveSchemaFieldLabel(t, subAgentsField.key, subAgentsField.label)
    : t(I18nKey.SCHEMA$ENABLE_SUB_AGENTS$LABEL);
  const subAgentsDescription = subAgentsField
    ? resolveSchemaFieldDescription(
        t,
        subAgentsField.key,
        subAgentsField.description,
      )
    : t(I18nKey.SCHEMA$ENABLE_SUB_AGENTS$DESCRIPTION);

  return (
    <div
      data-testid="agent-settings-screen"
      className="flex flex-col gap-6 pb-8 max-w-2xl"
    >
      {!embedded && (
        <div>
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$NAV_AGENT)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#A3A3A3]">
            {t(I18nKey.SETTINGS$AGENT_PAGE_DESCRIPTION)}
          </Typography.Paragraph>
        </div>
      )}

      <SettingsDropdownInput
        testId="agent-type-selector"
        name="agent-type"
        label={t(I18nKey.SETTINGS$NAV_AGENT)}
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
          const newType = key as AgentType;
          setAgentType(newType);
          if (newType === "acp" && !commandText) {
            const preferred = ACP_PROVIDERS[0];
            if (preferred) {
              setCommandText(formatCommand(preferred.default_command));
              setAcpModel(getAcpPreferredDefaultModel(preferred.key) ?? "");
              setIsCustomAcpModel(false);
            }
          } else if (newType === "openhands") {
            setIsCustomAcpModel(false);
          }
          setIsDirty(true);
        }}
      />

      {!isAcp && (
        <div className="flex flex-col gap-1.5">
          <SettingsSwitch
            testId="agent-settings-enable-sub-agents"
            isToggled={subAgentsEnabled}
            onToggle={(val) => {
              setSubAgentsEnabled(val);
            }}
          >
            {subAgentsLabel}
          </SettingsSwitch>
          {subAgentsDescription ? (
            <Typography.Paragraph
              className={cn(
                formControlSwitchDescriptionClassName,
                "text-tertiary-alt text-xs leading-5",
              )}
            >
              {subAgentsDescription}
            </Typography.Paragraph>
          ) : null}
        </div>
      )}

      {!isAcp && toolConcurrencyField ? (
        <SchemaField
          field={toolConcurrencyField}
          value={toolConcurrency}
          isDisabled={isSavingAny}
          onChange={setToolConcurrency}
        />
      ) : null}

      {isAcp && (
        <>
          <SettingsDropdownInput
            testId="agent-preset-selector"
            name="agent-preset"
            label={t(I18nKey.SETTINGS$AGENT_PRESET)}
            items={[
              ...ACP_PROVIDERS.map((provider) => ({
                key: provider.key,
                label: provider.display_name,
              })),
              {
                key: ACP_CUSTOM_PRESET_KEY,
                label: t(I18nKey.SETTINGS$AGENT_PRESET_CUSTOM),
              },
            ]}
            selectedKey={selectedPreset}
            onSelectionChange={(key) => {
              if (!key) return;
              const preset = String(key);
              const provider = getAcpProvider(preset);
              if (provider) {
                setCommandText(formatCommand(provider.default_command));
                setAcpModel(getAcpPreferredDefaultModel(preset) ?? "");
                setIsCustomAcpModel(false);
              } else if (preset === ACP_CUSTOM_PRESET_KEY) {
                // Clear command + model: the previous provider's default
                // command would otherwise make detectPreset(commandText)
                // re-match it on the next render and snap the dropdown back
                // off "Custom". Clearing model also prevents leaking e.g.
                // ``claude-opus-4-7`` into ``acp_model`` for an unrelated
                // wrapper.
                setCommandText("");
                setAcpModel("");
                setIsCustomAcpModel(true);
              }
              setIsDirty(true);
            }}
          />

          <div className="flex flex-col gap-2.5">
            <Typography.Text className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_COMMAND)}
            </Typography.Text>
            <textarea
              data-testid="agent-command-input"
              className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-white placeholder:text-[#717888] min-h-[60px] resize-y focus:outline-none focus:border-white"
              value={commandText}
              placeholder={commandPlaceholder}
              onChange={(e) => {
                const nextCommandText = e.target.value;
                // Keep the model selector in sync with the command being
                // typed. Editing the command into a *different* provider — or
                // into a custom command — must drop the previous provider's
                // model, or Save would silently persist e.g.
                // ``claude-opus-4-7`` against a Codex / custom wrapper. The
                // preset dropdown already does this; the textarea is the other
                // way a user changes provider, so it needs the same
                // reconciliation. Gated on the *detected preset* actually
                // changing, so it never clobbers a model the user is editing
                // within the same provider.
                const prevPreset = detectPreset(commandText, ACP_PROVIDERS);
                const nextPreset = detectPreset(nextCommandText, ACP_PROVIDERS);
                if (nextPreset !== prevPreset) {
                  setAcpModel(getAcpPreferredDefaultModel(nextPreset) ?? "");
                  setIsCustomAcpModel(false);
                }
                setCommandText(nextCommandText);
                setIsDirty(true);
              }}
            />
            <Typography.Text className="text-xs text-[#717888]">
              {t(I18nKey.SETTINGS$AGENT_COMMAND_HINT)}
            </Typography.Text>
          </div>

          <div className="flex flex-col gap-1.5">
            {hasModelSuggestions && (
              <SettingsDropdownInput
                testId="agent-model-selector"
                name="agent-model"
                label={t(I18nKey.SETTINGS$AGENT_MODEL)}
                items={[
                  ...modelSuggestions.map((model) => ({
                    key: model.id,
                    label: model.label,
                  })),
                  {
                    key: ACP_CUSTOM_MODEL_KEY,
                    label: t(I18nKey.SETTINGS$AGENT_PRESET_CUSTOM),
                  },
                ]}
                selectedKey={selectedModelKey}
                onSelectionChange={(key) => {
                  if (!key) return;
                  const modelKey = String(key);
                  if (modelKey === ACP_CUSTOM_MODEL_KEY) {
                    setIsCustomAcpModel(true);
                    setAcpModel("");
                  } else {
                    setIsCustomAcpModel(false);
                    setAcpModel(modelKey);
                  }
                  setIsDirty(true);
                }}
              />
            )}
            {selectedModelKey === ACP_CUSTOM_MODEL_KEY && (
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
        </>
      )}

      {isAcp && selectedPreset !== ACP_CUSTOM_PRESET_KEY && (
        <>
          <hr className="border-[#3D4046]" />
          <AcpCredentialsSection
            form={acpCredentialForm}
            providerKey={selectedPreset}
          />
        </>
      )}

      {!embedded && (
        <div>
          <BrandButton
            testId="agent-save-button"
            type="button"
            variant="primary"
            isDisabled={isSavingAny || !isAnyDirty || isAcpInvalid}
            onClick={handleSave}
          >
            {isSavingAny
              ? t(I18nKey.SETTINGS$SAVING)
              : t(I18nKey.SETTINGS$SAVE_CHANGES)}
          </BrandButton>
        </div>
      )}
    </div>
  );
}

/**
 * Legacy `/settings/agent` route. Settings → Agent is now the Agent Profile
 * library (`/settings/agents`), whose editor reuses the named
 * `AgentSettingsScreen` export below; this global-agent-form route is retired
 * and redirects there so old links/bookmarks keep working.
 *
 * Note: This is a route file; only the router should import the default export.
 * React Router's Vite plugin wraps a route's default export with
 * `withComponentProps`, which invokes it with route props and drops any props
 * passed by a parent — so embedded consumers (the Agent-profile editor) MUST
 * import the named `AgentSettingsScreen` export instead, or `embedded` /
 * `onSaveControlChange` never arrive. Mirrors `LlmSettingsRoute`.
 */
export default function AgentSettingsRoute() {
  return <Navigate to="/settings/agents" replace />;
}
