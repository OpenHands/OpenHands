import React, { useCallback, useEffect, useRef, useState } from "react";
import { Navigate } from "react-router";
import { useTranslation } from "react-i18next";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  buildProfileToolsValue,
  readProfileTools,
  type ProfileToolSpec,
} from "#/constants/profile-tools";
import { ProfileScopeList } from "#/components/features/settings/agent-profiles/profile-scope-list";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { SettingsFieldSchema, SettingsValue } from "#/types/settings";
import {
  coerceFieldValue,
  normalizeFieldValue,
} from "#/utils/sdk-settings-schema";
import {
  ACP_CUSTOM_PRESET_KEY,
  getAcpPreferredDefaultModel,
  getAcpProvider,
  getAcpProvidersForBackend,
  getAcpProviderSecrets,
  type ACPProviderConfig,
} from "#/constants/acp-providers";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { parseCommand, formatCommand } from "#/utils/acp-command";
import {
  readProfileScope,
  sameScopeSelection,
  type ProfileScopeMode,
} from "#/constants/profile-scope";
import { useToolCatalog } from "#/hooks/query/use-tool-catalog";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import { parseMcpConfig } from "#/utils/mcp-config";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsInstructions,
  agentProfileSupportsTools,
} from "#/api/agent-profiles-service/profile-field-support";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";

export const handle = { hideTitle: true };

type AgentType = "openhands" | "acp";

type AgentSettingsSnapshot = {
  agentType: AgentType;
  commandText: string;
  acpModel: string;
  isCustomAcpModel: boolean;
};

const TOOL_CONCURRENCY_FIELD_KEY = "tool_concurrency_limit";
const MCP_SERVER_REFS_KEY = "mcp_server_refs";
const SECRET_REFS_KEY = "secret_refs";
const TOOLS_KEY = "tools";
const PERSONA_KEY = "persona";
const SYSTEM_MESSAGE_SUFFIX_KEY = "system_message_suffix";
const SYSTEM_PROMPT_MAX_LENGTH = 65536;
const COMMAND_PLACEHOLDER_FALLBACK = "npx -y <package-name>";
const ACP_CUSTOM_MODEL_KEY = "__custom_model__";
const EMPTY_AGENT_SETTINGS_SNAPSHOT: AgentSettingsSnapshot = {
  agentType: "openhands",
  commandText: "",
  acpModel: "",
  isCustomAcpModel: false,
};

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

export type SystemPromptMode = "standard" | "append" | "custom";

/** A stored persona wins over stored instructions; the editor models one choice. */
export function readSystemPromptSeed(
  override: Record<string, SettingsValue> | null | undefined,
): { mode: SystemPromptMode; text: string } {
  const persona = override?.[PERSONA_KEY];
  if (typeof persona === "string" && persona)
    return { mode: "custom", text: persona };
  const suffix = override?.[SYSTEM_MESSAGE_SUFFIX_KEY];
  if (typeof suffix === "string" && suffix)
    return { mode: "append", text: suffix };
  return { mode: "standard", text: "" };
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
      mcp_server_refs: string[] | null;
      tool_concurrency_limit?: number;
      secret_refs?: string[] | null;
      tools?: ProfileToolSpec[] | null;
      persona?: string | null;
      system_message_suffix?: string | null;
    }
  | {
      agent_kind: "acp";
      mcp_server_refs: string[] | null;
      secret_refs?: string[] | null;
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
  toolConcurrencyField?: SettingsFieldSchema;
  toolConcurrency: string | boolean;
  mcpMode: ProfileScopeMode;
  selectedMcpServers: string[];
  secretsMode: ProfileScopeMode;
  selectedSecrets: string[];
  /** Whether the backend's *profile* model accepts `secret_refs`. */
  secretRefsSupportedOnProfile: boolean;
  toolsMode?: ProfileScopeMode;
  selectedTools?: string[];
  /** Stored params per tool, kept so the editor cannot drop what it ignores. */
  toolParams?: Record<string, Record<string, SettingsValue>>;
  /** Whether the tool catalog loaded; `tools` is only written then. */
  toolCatalogLoaded?: boolean;
  systemPromptMode: SystemPromptMode;
  systemPromptText: string;
  /** False when the section is hidden, so neither prompt field is written. */
  systemPromptEditable: boolean;
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
    toolConcurrencyField,
    toolConcurrency,
    mcpMode,
    selectedMcpServers,
    secretsMode,
    selectedSecrets,
    secretRefsSupportedOnProfile,
    toolsMode = "standard",
    selectedTools = [],
    toolParams = {},
    toolCatalogLoaded = false,
    systemPromptMode,
    systemPromptText,
    systemPromptEditable,
  } = input;
  // Both are base-model fields, so they ride both variants. `mcp_server_refs`
  // needs no version gate — it has existed since agent profiles shipped, below
  // the supported floor — while `secret_refs` does.
  const mcpRefs = {
    mcp_server_refs: mcpMode === "custom" ? selectedMcpServers : null,
  };
  const secretRefs = secretRefsSupportedOnProfile
    ? { secret_refs: secretsMode === "custom" ? selectedSecrets : null }
    : {};
  if (isAcp) {
    const isBuiltinDefault =
      isDefaultProviderCommand && selectedPreset !== ACP_CUSTOM_PRESET_KEY;
    return {
      agent_kind: "acp",
      ...mcpRefs,
      ...secretRefs,
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
      ...mcpRefs,
      ...secretRefs,
    };
  if (systemPromptEditable) {
    const text = systemPromptText.trim() ? systemPromptText : null;
    fields.system_message_suffix = systemPromptMode === "append" ? text : null;
    fields.persona = systemPromptMode === "custom" ? text : null;
  }
  if (toolCatalogLoaded) {
    fields.tools = buildProfileToolsValue({
      mode: toolsMode,
      selected: selectedTools,
      params: toolParams,
    });
  }
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
 * Handle the form exposes to the Agent-profile editor so it can read the
 * current state and persist it as an AgentProfile.
 */
export interface AgentSettingsSaveControl {
  agentType: AgentType;
  /** False when the current form can't be saved (e.g. an empty ACP command). */
  isValid: boolean;
  /** True when the form differs from the hydrated snapshot. */
  isDirty: boolean;
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
  /** `agent_settings`-shaped profile fields the form opens on. */
  agentSettingsOverride?: Record<string, SettingsValue> | null;
  onSaveControlChange: (control: AgentSettingsSaveControl) => void;
  /** Local launches route the `default` profile around its stored prompt and tools. */
  isDefaultProfile?: boolean;
}

export function AgentSettingsScreen({
  agentSettingsOverride = null,
  onSaveControlChange,
  isDefaultProfile = false,
}: AgentSettingsScreenProps) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading } = useSettings();
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const acpProviders = getAcpProvidersForBackend(
    useActiveBackend().backend.kind,
  );

  const fields = React.useMemo(
    () => schema?.sections.flatMap((section) => section.fields),
    [schema],
  );

  // --- Parallel tool calls (OpenHands path) ---
  // Surfaced only when the backend schema exposes the field, so older
  // agent-servers that predate ``tool_concurrency_limit`` hide it cleanly.
  const toolConcurrencyField = fields?.find(
    (field) => field.key === TOOL_CONCURRENCY_FIELD_KEY,
  );
  const initialToolConcurrency = React.useMemo(() => {
    if (!toolConcurrencyField) return "";
    const raw = agentSettingsOverride?.[TOOL_CONCURRENCY_FIELD_KEY];
    return normalizeFieldValue(toolConcurrencyField, raw);
  }, [toolConcurrencyField, agentSettingsOverride]);
  const [toolConcurrency, setToolConcurrency] = useState<string | boolean>(
    initialToolConcurrency,
  );

  // --- Tools (OpenHands only) ---
  const initialTools = React.useMemo(
    () => readProfileTools(agentSettingsOverride?.[TOOLS_KEY]),
    [agentSettingsOverride],
  );
  const [toolsMode, setToolsMode] = useState<ProfileScopeMode>(
    initialTools.mode,
  );
  const [selectedTools, setSelectedTools] = useState<string[] | null>(
    initialTools.mode === "custom" ? initialTools.selected : null,
  );
  const toolsSupported = agentProfileSupportsTools();
  const toolsEditable = toolsSupported && !isDefaultProfile;
  const {
    data: toolCatalog,
    isError: toolCatalogFailed,
    refetch: refetchToolCatalog,
  } = useToolCatalog({ enabled: toolsEditable });
  // Only a loaded catalog may write `tools`.
  const toolCatalogLoaded = toolCatalog !== undefined;
  const standardToolNames = React.useMemo(
    () =>
      toolCatalog
        ?.filter(({ in_default_set: inDefault, usable }) => inDefault && usable)
        .map(({ name }) => name),
    [toolCatalog],
  );
  /** The server's blurb for a tool, by name. */
  const toolDescriptions = React.useMemo(
    () =>
      new Map(
        (toolCatalog ?? []).map(({ name, description }) => [name, description]),
      ),
    [toolCatalog],
  );
  /** Pickable tools this runtime can run, plus anything the profile stores. */
  const toolPickerCatalog = React.useMemo(() => {
    const items: { name: string; description?: string }[] = (toolCatalog ?? [])
      .filter(({ user_selectable: selectable, usable }) => selectable && usable)
      .map(({ name, description }) => ({ name, description }));
    // Stored names outside the catalog stay selectable so they are kept.
    initialTools.selected.forEach((name) => {
      if (!items.some((item) => item.name === name))
        items.push({ name, description: toolDescriptions.get(name) });
    });
    return items;
  }, [toolCatalog, toolDescriptions, initialTools]);
  const orderedSelectedTools = React.useMemo(
    () =>
      toolPickerCatalog
        .map(({ name }) => name)
        .filter((name) => selectedTools?.includes(name)),
    [toolPickerCatalog, selectedTools],
  );

  // --- System prompt (OpenHands path) ---
  const instructionsSupported = agentProfileSupportsInstructions();
  const showSystemPrompt = instructionsSupported && !isDefaultProfile;
  const initialSystemPrompt = React.useMemo(
    () => readSystemPromptSeed(agentSettingsOverride),
    [agentSettingsOverride],
  );
  const [systemPromptMode, setSystemPromptMode] = useState<SystemPromptMode>(
    initialSystemPrompt.mode,
  );
  const [systemPromptText, setSystemPromptText] = useState(
    initialSystemPrompt.text,
  );

  // --- MCP servers (both variants; a base-model field) ---
  const initialMcpRefs = React.useMemo(
    () => readProfileScope(agentSettingsOverride?.[MCP_SERVER_REFS_KEY]),
    [agentSettingsOverride],
  );
  const [mcpMode, setMcpMode] = useState<ProfileScopeMode>(initialMcpRefs.mode);
  const [selectedMcpServers, setSelectedMcpServers] = useState<string[]>(
    initialMcpRefs.selected,
  );
  const configuredMcpNames = React.useMemo(
    () =>
      flattenMcpConfig(
        settings?.mcp_config ??
          parseMcpConfig(settings?.agent_settings?.mcp_config),
      )
        // `name` is optional on the shared type; flattenMcpConfig always sets
        // it from the config key, so narrow rather than assert.
        .filter(
          (server): server is typeof server & { name: string } =>
            typeof server.name === "string",
        )
        .map((server) => ({ name: server.name, description: server.type })),
    [settings],
  );
  const mcpCatalog = React.useMemo(() => {
    const configured = configuredMcpNames;
    // A stored ref whose server is gone rides along so the user can clear it:
    // a dangling ref fails the launch (422 locally; on cloud the never-brick
    // handler swallows it and silently falls back to unscoped settings).
    const known = new Set(configured.map(({ name }) => name));
    return [
      ...configured,
      ...initialMcpRefs.selected
        .filter((name) => !known.has(name))
        .map((name) => ({ name, description: null as string | null })),
    ];
  }, [configuredMcpNames, initialMcpRefs]);
  const orderedSelectedMcpServers = React.useMemo(
    () =>
      mcpCatalog
        .map(({ name }) => name)
        .filter((name) => selectedMcpServers.includes(name)),
    [mcpCatalog, selectedMcpServers],
  );
  // --- Secret scope (both variants; a base-model field) ---
  const secretRefsSupportedOnProfile = agentProfileSupportsSecretRefs();
  const { data: savedSecrets } = useSearchSecrets({
    enabled: secretRefsSupportedOnProfile,
  });
  const initialSecretRefs = React.useMemo(
    () => readProfileScope(agentSettingsOverride?.[SECRET_REFS_KEY]),
    [agentSettingsOverride],
  );
  const [secretsMode, setSecretsMode] = useState<ProfileScopeMode>(
    initialSecretRefs.mode,
  );
  const [selectedSecrets, setSelectedSecrets] = useState<string[]>(
    initialSecretRefs.selected,
  );
  // Scoping is strict server-side — nothing is added back — so an ACP profile
  // that omits its provider credential simply fails to authenticate. Keep the
  // names selected by default rather than re-adding them behind the user's
  // back: a visible checkbox they can still clear, not a hidden rule.
  const addProviderSecrets = useCallback(
    (providerKey: string) =>
      setSelectedSecrets((prev) => {
        const names = getAcpProviderSecrets(providerKey).map(
          ({ name }) => name,
        );
        const missing = names.filter((name) => !prev.includes(name));
        return missing.length ? [...prev, ...missing] : prev;
      }),
    [],
  );

  // A ref to a server that no longer exists 422s the launch, so surface it
  // while the user can still fix it.
  const danglingMcpRefs = React.useMemo(
    () =>
      mcpMode === "custom"
        ? orderedSelectedMcpServers.filter(
            (name) =>
              !configuredMcpNames.some((server) => server.name === name),
          )
        : [],
    [mcpMode, orderedSelectedMcpServers, configuredMcpNames],
  );

  // --- ACP path ---
  const [agentType, setAgentType] = useState<AgentType>("openhands");
  const [commandText, setCommandText] = useState("");
  const [acpModel, setAcpModel] = useState("");
  const [isCustomAcpModel, setIsCustomAcpModel] = useState(false);

  // ACP credentials live alongside the agent spec, so the form owns the
  // credential form and hands it to the editor's Save. Called unconditionally
  // (no-ops to empty fields for a non-ACP / custom command) to keep hook order
  // stable across the ``isLoading`` early-return below; ``detectPreset`` is a
  // cheap pure lookup.
  const acpPresetForCreds =
    agentType === "acp" ? detectPreset(commandText, acpProviders) : null;
  const acpCredentialForm = useAcpCredentialForm(
    acpPresetForCreds && acpPresetForCreds !== ACP_CUSTOM_PRESET_KEY
      ? acpPresetForCreds
      : null,
  );

  // Available provider credentials must remain visible when deselected, even
  // before their values are saved. Stored references also survive deletion.
  const secretCatalog = React.useMemo(() => {
    const saved = (savedSecrets ?? []).map((secret) => ({
      name: secret.name,
      description: secret.description ?? null,
    }));
    const known = new Set(saved.map((secret) => secret.name));
    return [
      ...saved,
      ...[
        ...new Set([
          ...initialSecretRefs.selected,
          ...acpCredentialForm.fields.map(({ name }) => name),
        ]),
      ]
        .filter((name) => !known.has(name))
        .map((name) => ({ name, description: null as string | null })),
    ];
  }, [savedSecrets, initialSecretRefs, acpCredentialForm.fields]);
  const orderedSelectedSecrets = React.useMemo(
    () =>
      secretCatalog
        .map(({ name }) => name)
        .filter((name) => selectedSecrets.includes(name)),
    [secretCatalog, selectedSecrets],
  );

  const [loadedSnapshot, setLoadedSnapshot] = useState<AgentSettingsSnapshot>(
    EMPTY_AGENT_SETTINGS_SNAPSHOT,
  );

  useEffect(() => {
    const kind = agentSettingsOverride?.agent_kind;

    if (kind === "acp") {
      setAgentType("acp");

      const rawAcpServer = agentSettingsOverride?.acp_server;
      const acpServer =
        typeof rawAcpServer === "string" ? rawAcpServer : undefined;
      const provider = getAcpProvider(acpServer);
      const storedCommand = toStringArray(agentSettingsOverride?.acp_command);
      const effectiveBaseCommand =
        storedCommand.length > 0
          ? storedCommand
          : (provider?.default_command ?? []);
      const tokens = [
        ...effectiveBaseCommand,
        ...toStringArray(agentSettingsOverride?.acp_args),
      ];
      const renderedCommandText =
        tokens.length > 0 ? formatCommand(tokens) : "";
      setCommandText(renderedCommandText);

      const savedModel = agentSettingsOverride?.acp_model;
      const normalizedSavedModel =
        typeof savedModel === "string" ? savedModel.trim() : "";
      const nextAcpModel =
        normalizedSavedModel || getAcpPreferredDefaultModel(acpServer) || "";
      const nextIsCustomAcpModel =
        !!normalizedSavedModel &&
        (!provider || !isKnownAcpModel(provider, normalizedSavedModel));
      setAcpModel(nextAcpModel);
      setIsCustomAcpModel(nextIsCustomAcpModel);
      setLoadedSnapshot({
        agentType: "acp",
        commandText: renderedCommandText,
        acpModel: nextAcpModel,
        isCustomAcpModel: nextIsCustomAcpModel,
      });
    } else {
      setAgentType("openhands");
      setCommandText("");
      setAcpModel("");
      setIsCustomAcpModel(false);
      setLoadedSnapshot(EMPTY_AGENT_SETTINGS_SNAPSHOT);
    }
  }, [agentSettingsOverride]);

  // Sync the parallel-tool-calls input when settings reload
  useEffect(() => {
    setToolConcurrency(initialToolConcurrency);
  }, [initialToolConcurrency]);

  useEffect(() => {
    setSystemPromptMode(initialSystemPrompt.mode);
    setSystemPromptText(initialSystemPrompt.text);
  }, [initialSystemPrompt]);

  // Sync the MCP scope when settings reload
  useEffect(() => {
    setMcpMode(initialMcpRefs.mode);
    setSelectedMcpServers(initialMcpRefs.selected);
  }, [initialMcpRefs]);

  // Seed a custom selection from the standard set once the catalog arrives,
  // without replacing edits made while it was pending.
  useEffect(() => {
    if (toolsMode !== "custom" || standardToolNames === undefined) return;
    setSelectedTools(
      (previous) =>
        previous ??
        standardToolNames.filter((name) =>
          toolPickerCatalog.some((entry) => entry.name === name),
        ),
    );
  }, [toolsMode, standardToolNames, toolPickerCatalog]);

  // Sync the tool selection when settings reload
  useEffect(() => {
    setToolsMode(initialTools.mode);
    setSelectedTools(
      initialTools.mode === "custom" ? initialTools.selected : null,
    );
  }, [initialTools]);

  // Sync the secret scope when settings reload
  useEffect(() => {
    setSecretsMode(initialSecretRefs.mode);
    setSelectedSecrets(initialSecretRefs.selected);
  }, [initialSecretRefs]);

  // --- Save control ---
  // Ref-backed so the exposed builder/credential fns read the freshest state at
  // call time without re-emitting the control on every keystroke (mirrors
  // ``sdk-section-page``). The body is (re)assigned during render below.
  const buildFieldsRef = useRef<() => AgentProfileFieldsDraft>(() => ({
    agent_kind: "openhands",
    mcp_server_refs: null,
  }));
  const stableBuildFields = useCallback(() => buildFieldsRef.current(), []);
  const credFormRef = useRef(acpCredentialForm);
  credFormRef.current = acpCredentialForm;
  const stableCredSave = useCallback(
    (opts?: { silent?: boolean }) => credFormRef.current.save(opts),
    [],
  );
  const stableCredReset = useCallback(() => credFormRef.current.reset(), []);

  // Validity/kind/dirty are computed here (before the loading early-return) so
  // the emit effect can depend on them; the full ACP derivation lives after it.
  const acpCommandEmpty =
    agentType === "acp" && parseCommand(commandText).length === 0;
  // `mcp_server_refs` lives on the profile base, so it is dirty-tracked for both
  // variants rather than inside the kind-specific branch below.
  const mcpScopeDirty =
    mcpMode !== initialMcpRefs.mode ||
    !sameScopeSelection(orderedSelectedMcpServers, initialMcpRefs.selected);
  // `secret_refs` is a base-model field too, so it is tracked for both variants.
  const secretScopeDirty =
    secretsMode !== initialSecretRefs.mode ||
    !sameScopeSelection(orderedSelectedSecrets, initialSecretRefs.selected);
  const toolSelectionDirty =
    toolsMode !== initialTools.mode ||
    (toolsMode === "custom" &&
      !sameScopeSelection(orderedSelectedTools, initialTools.selected));
  const systemPromptDirty =
    showSystemPrompt &&
    (systemPromptMode !== initialSystemPrompt.mode ||
      (systemPromptMode !== "standard" &&
        systemPromptText !== initialSystemPrompt.text));
  const systemPromptMissing =
    agentType === "openhands" &&
    showSystemPrompt &&
    systemPromptMode !== "standard" &&
    !systemPromptText.trim();
  const settingsDirty =
    agentType !== loadedSnapshot.agentType ||
    mcpScopeDirty ||
    secretScopeDirty ||
    toolSelectionDirty ||
    (agentType === "acp"
      ? commandText !== loadedSnapshot.commandText ||
        acpModel !== loadedSnapshot.acpModel ||
        isCustomAcpModel !== loadedSnapshot.isCustomAcpModel
      : toolConcurrency !== initialToolConcurrency || systemPromptDirty);
  const credentialsDirty = acpCredentialForm.isDirty;
  const isAnyDirty = settingsDirty || credentialsDirty;
  const customToolsUninitialized =
    toolCatalogLoaded &&
    agentType === "openhands" &&
    toolsMode === "custom" &&
    selectedTools === null;
  const isFormValid =
    !acpCommandEmpty && !systemPromptMissing && !customToolsUninitialized;
  useEffect(() => {
    onSaveControlChange({
      agentType,
      isValid: isFormValid,
      isDirty: isAnyDirty,
      buildAgentProfileFields: stableBuildFields,
      credentials: {
        isDirty: credentialsDirty,
        save: stableCredSave,
        reset: stableCredReset,
      },
    });
  }, [
    onSaveControlChange,
    agentType,
    isFormValid,
    isAnyDirty,
    credentialsDirty,
    stableBuildFields,
    stableCredSave,
    stableCredReset,
  ]);

  if (isLoading) return null;

  const isAcp = agentType === "acp";
  const commandTokens = parseCommand(commandText);
  const selectedPreset = detectPreset(commandText, acpProviders);
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
    formatCommand(acpProviders[0]?.default_command ?? []) ||
    COMMAND_PLACEHOLDER_FALLBACK;

  // Assign the control's field builder from the live render state.
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
      toolConcurrencyField,
      toolConcurrency,
      mcpMode,
      selectedMcpServers: orderedSelectedMcpServers,
      secretsMode,
      selectedSecrets: orderedSelectedSecrets,
      secretRefsSupportedOnProfile,
      toolsMode,
      selectedTools: orderedSelectedTools,
      toolParams: initialTools.params,
      toolCatalogLoaded,
      systemPromptMode,
      systemPromptText,
      systemPromptEditable: showSystemPrompt,
    });

  const isSaving = acpCredentialForm.isSaving;

  return (
    <div
      data-testid="agent-settings-screen"
      className="flex flex-col gap-6 pb-8 max-w-2xl"
    >
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
            const preferred = acpProviders[0];
            if (preferred) {
              setCommandText(formatCommand(preferred.default_command));
              setAcpModel(getAcpPreferredDefaultModel(preferred.key) ?? "");
              setIsCustomAcpModel(false);
              if (secretsMode === "custom") addProviderSecrets(preferred.key);
            }
          } else if (
            newType === "acp" &&
            agentType !== "acp" &&
            secretsMode === "custom"
          ) {
            addProviderSecrets(selectedPreset);
          } else if (newType === "openhands") {
            setIsCustomAcpModel(false);
          }
        }}
      />

      {!isAcp && instructionsSupported ? (
        <div className="flex flex-col gap-2.5">
          <Typography.Text className="text-sm">
            {t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT)}
          </Typography.Text>
          {isDefaultProfile ? (
            <Typography.Text
              testId="agent-settings-system-prompt-default-profile"
              className="text-xs text-tertiary-alt"
            >
              {t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_DEFAULT_PROFILE)}
            </Typography.Text>
          ) : (
            <>
              <SettingsDropdownInput
                testId="agent-settings-system-prompt-mode"
                name="agent-system-prompt-mode"
                label=""
                items={[
                  {
                    key: "standard",
                    label: t(
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_STANDARD,
                    ),
                  },
                  {
                    key: "append",
                    label: t(
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND,
                    ),
                  },
                  {
                    key: "custom",
                    label: t(
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM,
                    ),
                  },
                ]}
                selectedKey={systemPromptMode}
                isDisabled={isSaving}
                onSelectionChange={(key) => {
                  if (key) setSystemPromptMode(key as SystemPromptMode);
                }}
              />
              {systemPromptMode !== "standard" ? (
                <textarea
                  data-testid="agent-settings-system-prompt"
                  aria-label={t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT)}
                  className={cn(
                    formControlMultilineFieldClassName,
                    "min-h-48 font-mono placeholder:italic",
                    "disabled:bg-surface-raised disabled:border-border-subtle",
                  )}
                  value={systemPromptText}
                  maxLength={SYSTEM_PROMPT_MAX_LENGTH}
                  placeholder={t(
                    systemPromptMode === "custom"
                      ? I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_PLACEHOLDER
                      : I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND_PLACEHOLDER,
                  )}
                  disabled={isSaving}
                  onChange={(e) => setSystemPromptText(e.target.value)}
                />
              ) : null}
              {systemPromptMissing ? (
                <Typography.Text
                  testId="agent-settings-system-prompt-required"
                  className="text-xs text-danger"
                >
                  {t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_REQUIRED)}
                </Typography.Text>
              ) : null}
              <Typography.Text
                testId="agent-settings-system-prompt-hint"
                className="text-xs text-tertiary-alt"
              >
                {t(
                  {
                    standard:
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_STANDARD_HINT,
                    append:
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND_HINT,
                    custom:
                      I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM_HINT,
                  }[systemPromptMode],
                )}
              </Typography.Text>
            </>
          )}
        </div>
      ) : null}

      {!isAcp && toolConcurrencyField ? (
        <SchemaField
          field={toolConcurrencyField}
          value={toolConcurrency}
          isDisabled={isSaving}
          onChange={setToolConcurrency}
        />
      ) : null}
      {!isAcp && toolsSupported && isDefaultProfile ? (
        <div className="flex flex-col gap-2.5">
          <Typography.Text className="text-sm">
            {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS)}
          </Typography.Text>
          <Typography.Text
            testId="agent-settings-tools-default-profile"
            className="text-xs text-tertiary-alt"
          >
            {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_DEFAULT_PROFILE)}
          </Typography.Text>
        </div>
      ) : null}
      {!isAcp && toolsEditable ? (
        <div className="flex flex-col gap-2.5">
          <Typography.Text className="text-sm">
            {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS)}
          </Typography.Text>
          <SettingsDropdownInput
            testId="agent-settings-tools-mode"
            name="agent-tools-mode"
            label=""
            items={[
              {
                key: "standard",
                label: t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_STANDARD),
              },
              {
                key: "custom",
                label: t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_CHOOSE),
              },
            ]}
            selectedKey={toolsMode}
            isDisabled={isSaving || standardToolNames === undefined}
            onSelectionChange={(key) => {
              if (!key) return;
              setToolsMode(key as ProfileScopeMode);
            }}
          />
          {toolCatalogFailed && standardToolNames === undefined ? (
            <div
              className="flex items-center gap-2"
              data-testid="agent-settings-tools-load-failed"
            >
              <Typography.Text className="text-xs text-red-400">
                {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_LOAD_FAILED)}{" "}
                {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_KEPT_ON_SAVE)}
              </Typography.Text>
              <BrandButton
                testId="agent-settings-tools-retry"
                type="button"
                variant="secondary"
                onClick={() => refetchToolCatalog()}
              >
                {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_RETRY)}
              </BrandButton>
            </div>
          ) : toolsMode === "standard" ? (
            <ProfileScopeList
              testId="agent-settings-tool"
              items={(standardToolNames ?? []).map((name) => ({
                name,
                description: toolDescriptions.get(name),
              }))}
              selected={standardToolNames ?? []}
              isDisabled
              onToggle={() => {}}
            />
          ) : (
            <ProfileScopeList
              testId="agent-settings-tool"
              items={toolPickerCatalog}
              selected={orderedSelectedTools}
              isDisabled={isSaving || !toolCatalogLoaded}
              onToggle={(name, checked) =>
                setSelectedTools((prev) =>
                  checked
                    ? [...(prev ?? []), name]
                    : (prev ?? []).filter((entry) => entry !== name),
                )
              }
            />
          )}
          <Typography.Text className="text-xs text-tertiary-alt">
            {t(
              toolsMode === "custom"
                ? I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_CHOOSE_HINT
                : I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_STANDARD_HINT,
            )}
          </Typography.Text>
        </div>
      ) : null}

      <div className="flex flex-col gap-2.5">
        <Typography.Text className="text-sm">
          {t(I18nKey.SETTINGS$AGENT_PROFILE_MCP)}
        </Typography.Text>
        <SettingsDropdownInput
          testId="agent-settings-mcp-mode"
          name="agent-mcp-mode"
          label=""
          items={[
            {
              key: "standard",
              label: t(I18nKey.SETTINGS$AGENT_PROFILE_MCP_ALL),
            },
            {
              key: "custom",
              label: t(I18nKey.SETTINGS$AGENT_PROFILE_MCP_CHOOSE),
            },
          ]}
          selectedKey={mcpMode}
          isDisabled={isSaving}
          onSelectionChange={(key) => {
            if (!key) return;
            const mode = key as ProfileScopeMode;
            setMcpMode(mode);
            // Seed a first switch to custom from the default — every
            // configured server — so turning the control on narrows from what
            // the agent had rather than cutting it off from all of them.
            if (mode === "custom" && selectedMcpServers.length === 0) {
              setSelectedMcpServers(configuredMcpNames.map(({ name }) => name));
            }
          }}
        />
        {mcpCatalog.length > 0 ? (
          <ProfileScopeList
            testId="agent-settings-mcp"
            items={mcpCatalog}
            selected={
              mcpMode === "custom"
                ? orderedSelectedMcpServers
                : mcpCatalog.map(({ name }) => name)
            }
            isDisabled={isSaving || mcpMode === "standard"}
            onToggle={(name, checked) =>
              setSelectedMcpServers((prev) =>
                checked
                  ? [...prev, name]
                  : prev.filter((entry) => entry !== name),
              )
            }
          />
        ) : (
          <Typography.Text className="text-xs text-tertiary-alt">
            {t(I18nKey.SETTINGS$AGENT_PROFILE_MCP_NONE)}
          </Typography.Text>
        )}
        {danglingMcpRefs.length > 0 ? (
          <Typography.Text
            testId="agent-settings-mcp-dangling"
            className="text-xs text-danger"
          >
            {t(I18nKey.SETTINGS$AGENT_PROFILE_MCP_DANGLING, {
              names: danglingMcpRefs.join(", "),
            })}
          </Typography.Text>
        ) : null}
        <Typography.Text className="text-xs text-tertiary-alt">
          {t(
            mcpMode === "custom"
              ? I18nKey.SETTINGS$AGENT_PROFILE_MCP_CHOOSE_HINT
              : I18nKey.SETTINGS$AGENT_PROFILE_MCP_ALL_HINT,
          )}
        </Typography.Text>
      </div>

      {secretRefsSupportedOnProfile ? (
        <div className="flex flex-col gap-2.5">
          <Typography.Text className="text-sm">
            {t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS)}
          </Typography.Text>
          <SettingsDropdownInput
            testId="agent-settings-secrets-mode"
            name="agent-secrets-mode"
            label=""
            items={[
              {
                key: "standard",
                label: t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS_ALL),
              },
              {
                key: "custom",
                label: t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS_CHOOSE),
              },
            ]}
            selectedKey={secretsMode}
            isDisabled={isSaving}
            onSelectionChange={(key) => {
              if (!key) return;
              const mode = key as ProfileScopeMode;
              setSecretsMode(mode);
              if (mode === "custom" && isAcp)
                addProviderSecrets(selectedPreset);
            }}
          />
          {secretCatalog.length > 0 ? (
            <ProfileScopeList
              testId="agent-settings-secret"
              items={secretCatalog}
              selected={
                secretsMode === "custom"
                  ? orderedSelectedSecrets
                  : secretCatalog.map(({ name }) => name)
              }
              isDisabled={isSaving || secretsMode === "standard"}
              onToggle={(name, checked) =>
                setSelectedSecrets((prev) =>
                  checked
                    ? [...prev, name]
                    : prev.filter((entry) => entry !== name),
                )
              }
            />
          ) : (
            <Typography.Text className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS_NONE)}
            </Typography.Text>
          )}
          <Typography.Text className="text-xs text-tertiary-alt">
            {t(
              secretsMode === "custom"
                ? I18nKey.SETTINGS$AGENT_PROFILE_SECRETS_CHOOSE_HINT
                : I18nKey.SETTINGS$AGENT_PROFILE_SECRETS_ALL_HINT,
            )}
          </Typography.Text>
        </div>
      ) : null}

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
                if (secretsMode === "custom" && preset !== selectedPreset) {
                  addProviderSecrets(preset);
                }
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
            }}
          />

          <div className="flex flex-col gap-2.5">
            <Typography.Text className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_COMMAND)}
            </Typography.Text>
            <textarea
              data-testid="agent-command-input"
              className="bg-tertiary border border-[#717888] rounded-sm p-2 text-sm font-mono text-contrast placeholder:text-[#717888] min-h-15 resize-y focus:outline-none focus:border-contrast"
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
                const prevPreset = detectPreset(commandText, acpProviders);
                const nextPreset = detectPreset(nextCommandText, acpProviders);
                if (nextPreset !== prevPreset) {
                  setAcpModel(getAcpPreferredDefaultModel(nextPreset) ?? "");
                  setIsCustomAcpModel(false);
                  if (secretsMode === "custom") addProviderSecrets(nextPreset);
                }
                setCommandText(nextCommandText);
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
 * passed by a parent — so the Agent-profile editor MUST import the named
 * `AgentSettingsScreen` export instead, or `agentSettingsOverride` /
 * `onSaveControlChange` never arrive. Mirrors `LlmSettingsRoute`.
 */
export default function AgentSettingsRoute() {
  return <Navigate to="/settings/agents" replace />;
}
