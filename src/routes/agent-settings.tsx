import React, { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Navigate } from "react-router";
import { useTranslation } from "react-i18next";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import {
  SdkSectionPage,
  type SdkSectionSaveControl,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import {
  fromBehaviorFormValues,
  GLOBAL_BEHAVIOR_FIELD_KEYS,
  toBehaviorFormValues,
  type ProfileBehaviorSection,
} from "#/components/features/settings/agent-profiles/profile-behavior-form";
import { AcpCredentialsSection } from "#/components/features/settings/acp-credentials-section";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  buildProfileToolsValue,
  readProfileTools,
  type ProfileToolSpec,
} from "#/constants/profile-tools";
import { ProfileScopeList } from "#/components/features/settings/agent-profiles/profile-scope-list";
import { AgentMcpSetup } from "#/components/features/settings/agent-profiles/agent-mcp-setup";
import { AgentFormSection } from "#/components/features/settings/agent-profiles/agent-form-section";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useDeleteSecret } from "#/hooks/mutation/use-delete-secret";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { AddSkillModal } from "#/components/features/skills/add-skill-modal";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import {
  Settings,
  SettingsFieldSchema,
  SettingsSchema,
  SettingsValue,
} from "#/types/settings";
import {
  coerceFieldValue,
  inferInitialView,
  normalizeFieldValue,
  type SettingsView,
} from "#/utils/sdk-settings-schema";
import {
  ACP_CUSTOM_PRESET_KEY,
  getAcpPreferredDefaultModel,
  getAcpProvider,
  getAcpProvidersForBackend,
  getAcpProviderSecrets,
  labelForAcpModel,
  type ACPModelOption,
  type ACPProviderConfig,
} from "#/constants/acp-providers";
import { useAcpModelDiscovery } from "#/hooks/query/use-acp-model-discovery";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { parseCommand, formatCommand } from "#/utils/acp-command";
import { enabledMcpServerRefs } from "#/hooks/mutation/sync-agent-profiles";
import {
  readProfileScope,
  sameScopeSelection,
  type ProfileScopeMode,
} from "#/constants/profile-scope";
import { useToolCatalog } from "#/hooks/query/use-tool-catalog";
import { useSkillCatalog } from "#/hooks/query/use-skill-catalog";
import {
  buildSkillEnablementFilter,
  profileSkillEnablement,
  publicSkillTest,
  sameSkillSelection,
  toggleSkill,
  toProfileSkillFields,
  type SkillEnablement,
} from "#/utils/skill-enablement";
import { useMetaProfiles } from "#/hooks/query/use-meta-profiles";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import { parseMcpConfig } from "#/utils/mcp-config";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsInstructions,
  agentProfileSupportsSkills,
  agentProfileSupportsTools,
  agentProfileSupportsRouter,
} from "#/api/agent-profiles-service/profile-field-support";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";

export const handle = { hideTitle: true };

const PROFILE_BEHAVIOR_BLOCKS: ReadonlyArray<{
  section: ProfileBehaviorSection;
  titleKey: I18nKey;
}> = [
  { section: "condenser", titleKey: I18nKey.SETTINGS$NAV_CONDENSER },
  { section: "verification", titleKey: I18nKey.SETTINGS$NAV_VERIFICATION },
];

type AgentType = "openhands" | "acp";

type AgentSettingsSnapshot = {
  agentType: AgentType;
  commandText: string;
  acpModel: string;
  isCustomAcpModel: boolean;
};

const TOOL_CONCURRENCY_FIELD_KEY = "tool_concurrency_limit";
// Not valid meta-profile names, so they can't collide with one.
const ROUTER_OFF_KEY = "__off__";
const ROUTER_FIRST_KEY = "__first__";
const SKILL_SEARCH_MIN_ITEMS = 10;
const MCP_SERVER_REFS_KEY = "mcp_server_refs";
const SECRET_REFS_KEY = "secret_refs";
const TOOLS_KEY = "tools";
const PERSONA_KEY = "persona";
const SYSTEM_MESSAGE_SUFFIX_KEY = "system_message_suffix";
const SYSTEM_PROMPT_MAX_LENGTH = 65536;
const COMMAND_PLACEHOLDER_FALLBACK = "npx -y <package-name>";
const ACP_CUSTOM_MODEL_KEY = "__custom_model__";
const ACP_AGENT_DEFAULT_MODEL_KEY = "__agent_default_model__";
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

/** The agent's models, keeping a saved model selectable when it isn't listed. */
function buildModelSuggestions(
  models: readonly ACPModelOption[],
  selectedModel: string,
): ACPModelOption[] {
  const model = selectedModel.trim();
  if (!model || models.some(({ id }) => id === model)) return [...models];
  return [...models, { id: model, label: model }];
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
      public_skills?: string[];
      disabled_skills?: string[];
      tools?: ProfileToolSpec[] | null;
      enable_classify_and_switch_llm_tool?: boolean;
      meta_profile_ref?: string | null;
      persona?: string | null;
      system_message_suffix?: string | null;
      condenser?: Record<string, SettingsValue>;
      verification?: Record<string, SettingsValue>;
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
  /** The skill selection to store; undefined keeps the stored one. */
  skillEnablement?: SkillEnablement;
  toolsMode?: ProfileScopeMode;
  selectedTools?: string[];
  /** Stored params per tool, kept so the editor cannot drop what it ignores. */
  toolParams?: Record<string, Record<string, SettingsValue>>;
  /** Whether the tool catalog loaded; `tools` is only written then. */
  toolCatalogLoaded?: boolean;
  /** The Model Router to store: a meta-profile name, null for none, undefined to keep it. */
  router?: string | null;
  systemPromptMode: SystemPromptMode;
  systemPromptText: string;
  /** False when the section is hidden, so neither prompt field is written. */
  systemPromptEditable: boolean;
  /** Profile sections to store; an absent one keeps the stored value. */
  condenser?: Record<string, SettingsValue>;
  verification?: Record<string, SettingsValue>;
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
    skillEnablement,
    toolsMode = "standard",
    selectedTools = [],
    toolParams = {},
    toolCatalogLoaded = false,
    router,
    systemPromptMode,
    systemPromptText,
    systemPromptEditable,
    condenser,
    verification,
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
  if (skillEnablement)
    Object.assign(fields, toProfileSkillFields(skillEnablement));
  if (systemPromptEditable) {
    const text = systemPromptText.trim() ? systemPromptText : null;
    fields.system_message_suffix = systemPromptMode === "append" ? text : null;
    fields.persona = systemPromptMode === "custom" ? text : null;
  }
  if (condenser) fields.condenser = condenser;
  if (verification) fields.verification = verification;
  if (toolCatalogLoaded) {
    fields.tools = buildProfileToolsValue({
      mode: toolsMode,
      selected: selectedTools,
      params: toolParams,
    });
  }
  if (router !== undefined) {
    fields.enable_classify_and_switch_llm_tool = router !== null;
    fields.meta_profile_ref = router;
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
  /** `default` shares its MCP servers and skills with the global pages and keeps every secret. */
  isDefaultProfile?: boolean;
  /** Required profile name, rendered beside the always-visible LLM selector. */
  profileNameField?: React.ReactNode;
  /** Shown under the agent type for an OpenHands agent. */
  llmPicker?: React.ReactNode;
}

export function AgentSettingsScreen({
  agentSettingsOverride = null,
  onSaveControlChange,
  isDefaultProfile = false,
  profileNameField = null,
  llmPicker = null,
}: AgentSettingsScreenProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const deleteSecret = useDeleteSecret();
  const { data: settings, isLoading } = useSettings();
  const { data: schema } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const backendKind = useActiveBackend().backend.kind;
  const acpProviders = getAcpProvidersForBackend(backendKind);

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
  const {
    data: toolCatalog,
    isError: toolCatalogFailed,
    refetch: refetchToolCatalog,
  } = useToolCatalog({ enabled: toolsSupported });
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
  const [addingSecret, setAddingSecret] = useState(false);
  const [showAddSkillHelp, setShowAddSkillHelp] = useState(false);
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
  // --- Skills (OpenHands; the selection a profile launch loads) ---
  const skillsSupported = agentProfileSupportsSkills();
  const {
    data: skillCatalog,
    isError: skillCatalogFailed,
    refetch: refetchSkillCatalog,
  } = useSkillCatalog({ enabled: skillsSupported });
  const isPublicSkill = React.useMemo(
    () => publicSkillTest(skillCatalog),
    [skillCatalog],
  );
  const [skillQuery, setSkillQuery] = useState("");
  const visibleSkills = React.useMemo(() => {
    const query = skillQuery.trim().toLowerCase();
    if (!skillCatalog || !query) return skillCatalog;
    return skillCatalog.filter((skill) =>
      [skill.name, skill.description ?? "", ...(skill.triggers ?? [])].some(
        (text) => text.toLowerCase().includes(query),
      ),
    );
  }, [skillCatalog, skillQuery]);
  const storedPublicSkills = agentSettingsOverride?.public_skills;
  const hasStoredSkillList = Array.isArray(storedPublicSkills);
  const initialSkills = React.useMemo<SkillEnablement | undefined>(() => {
    const disabled = agentSettingsOverride?.disabled_skills;
    const stored = {
      public_skills: hasStoredSkillList
        ? (storedPublicSkills as string[])
        : null,
      disabled_skills: Array.isArray(disabled) ? (disabled as string[]) : [],
    };
    if (hasStoredSkillList) return profileSkillEnablement(stored);
    // No stored list means every catalog skill, which only the catalog can name.
    if (!skillCatalog) return undefined;
    return profileSkillEnablement(
      stored,
      skillCatalog
        .filter((skill) => skill.origin === "public")
        .map((skill) => skill.name),
    );
  }, [
    agentSettingsOverride,
    hasStoredSkillList,
    storedPublicSkills,
    skillCatalog,
  ]);
  const [editedSkills, setEditedSkills] = useState<SkillEnablement | null>(
    null,
  );
  const currentSkills = editedSkills ?? initialSkills;
  const enabledSkillNames = React.useMemo(() => {
    if (!skillCatalog || !currentSkills) return [];
    const isEnabled = buildSkillEnablementFilter(currentSkills, isPublicSkill);
    return skillCatalog
      .map((skill) => skill.name)
      .filter((name) => isEnabled(name));
  }, [skillCatalog, currentSkills, isPublicSkill]);
  const toggleProfileSkill = useCallback(
    (name: string, enabled: boolean) =>
      setEditedSkills((previous) => {
        const base = previous ?? initialSkills;
        return base
          ? toggleSkill(base, name, enabled, isPublicSkill)
          : previous;
      }),
    [initialSkills, isPublicSkill],
  );
  // A seeded or stored list is written back; an unset one only once edited.
  const skillsToSave = skillsSupported
    ? (editedSkills ?? (hasStoredSkillList ? initialSkills : undefined))
    : undefined;
  const skillsDirty =
    editedSkills !== null &&
    initialSkills !== undefined &&
    !sameSkillSelection(editedSkills, initialSkills);

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
  const [secretToDelete, setSecretToDelete] = useState<string | null>(null);
  const [deletedSecrets, setDeletedSecrets] = useState<string[]>([]);
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
    const saved = (savedSecrets ?? [])
      .filter((secret) => !deletedSecrets.includes(secret.name))
      .map((secret) => ({
        name: secret.name,
        description: secret.description ?? null,
      }));
    const known = new Set(saved.map((secret) => secret.name));
    return [
      ...saved,
      ...[
        ...new Set([
          ...initialSecretRefs.selected,
          ...selectedSecrets,
          ...acpCredentialForm.fields.map(({ name }) => name),
        ]),
      ]
        .filter((name) => !known.has(name) && !deletedSecrets.includes(name))
        .map((name) => ({ name, description: null as string | null })),
    ];
  }, [
    savedSecrets,
    initialSecretRefs,
    selectedSecrets,
    deletedSecrets,
    acpCredentialForm.fields,
  ]);
  const orderedSelectedSecrets = React.useMemo(
    () =>
      secretCatalog
        .map(({ name }) => name)
        .filter((name) => selectedSecrets.includes(name)),
    [secretCatalog, selectedSecrets],
  );

  const handleConfirmDeleteSecret = async () => {
    if (!secretToDelete) return;
    try {
      await deleteSecret.mutateAsync(secretToDelete);
      setSelectedSecrets((current) =>
        current.filter((name) => name !== secretToDelete),
      );
      setDeletedSecrets((current) => [...current, secretToDelete]);
      void queryClient.invalidateQueries({ queryKey: ["secrets"] });
      setSecretToDelete(null);
    } catch (error) {
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    }
  };

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
      const nextIsCustomAcpModel = !!normalizedSavedModel && !provider;
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

  // --- Condenser and verification (OpenHands; stored on the profile) ---
  const [behaviorControls, setBehaviorControls] = useState<
    Partial<Record<ProfileBehaviorSection, SdkSectionSaveControl>>
  >({});
  const behaviorControlHandlers = React.useMemo(
    () =>
      Object.fromEntries(
        PROFILE_BEHAVIOR_BLOCKS.map(({ section }) => [
          section,
          (control: SdkSectionSaveControl) =>
            setBehaviorControls((previous) => ({
              ...previous,
              [section]: control,
            })),
        ]),
      ) as Record<
        ProfileBehaviorSection,
        (control: SdkSectionSaveControl) => void
      >,
    [],
  );
  // A new profile has no stored sections and starts from the global values.
  const storedBehavior = React.useMemo(() => {
    const asRecord = (value: unknown) =>
      value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
    return {
      condenser: asRecord(agentSettingsOverride?.condenser),
      verification: asRecord(agentSettingsOverride?.verification),
    };
  }, [agentSettingsOverride]);
  const behaviorInitialValues = React.useMemo(
    () => ({
      condenser: storedBehavior.condenser
        ? toBehaviorFormValues(schema, "condenser", storedBehavior.condenser)
        : undefined,
      verification: storedBehavior.verification
        ? toBehaviorFormValues(
            schema,
            "verification",
            storedBehavior.verification,
          )
        : undefined,
    }),
    [schema, storedBehavior],
  );
  // Open each section on the view its stored values need, not the global one.
  const behaviorInitialView = React.useMemo(
    () =>
      Object.fromEntries(
        PROFILE_BEHAVIOR_BLOCKS.map(({ section }) => {
          const stored = storedBehavior[section];
          return [
            section,
            stored
              ? (current: Settings, filteredSchema: SettingsSchema) =>
                  inferInitialView(
                    {
                      ...current,
                      agent_settings: {
                        ...current.agent_settings,
                        [section]: stored as SettingsValue,
                      },
                    },
                    filteredSchema,
                  )
              : undefined,
          ];
        }),
      ) as Record<
        ProfileBehaviorSection,
        | ((current: Settings, filteredSchema: SettingsSchema) => SettingsView)
        | undefined
      >,
    [storedBehavior],
  );
  const behaviorDirty = Boolean(
    behaviorControls.condenser?.isDirty ||
    behaviorControls.verification?.isDirty,
  );
  const behaviorToSave = (section: ProfileBehaviorSection) => {
    const control = behaviorControls[section];
    if (!control) return undefined;
    if (!control.isDirty && storedBehavior[section]) return undefined;
    return fromBehaviorFormValues(schema, section, control.values);
  };
  // --- Model Router (OpenHands; stored on the profile) ---
  const routerSupported = agentProfileSupportsRouter();
  const { data: metaProfilesData } = useMetaProfiles({
    enabled: routerSupported,
  });
  // On without a name means the server picks the alphabetically first router.
  let initialRouter: string | null = null;
  if (agentSettingsOverride?.enable_classify_and_switch_llm_tool === true) {
    initialRouter =
      typeof agentSettingsOverride.meta_profile_ref === "string"
        ? agentSettingsOverride.meta_profile_ref
        : ROUTER_FIRST_KEY;
  }
  const [router, setRouter] = useState<string | null>(initialRouter);
  useEffect(() => {
    setRouter(initialRouter);
  }, [initialRouter]);
  const routerNames = (metaProfilesData?.meta_profiles ?? []).map(
    ({ name }) => name,
  );
  // A router that no longer exists fails the launch, so keep it visible.
  const routerDangling =
    router !== null &&
    router !== ROUTER_FIRST_KEY &&
    !!metaProfilesData &&
    !routerNames.includes(router);

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
  const routerDirty = routerSupported && router !== initialRouter;
  const toolSelectionDirty =
    toolsMode !== initialTools.mode ||
    (toolsMode === "custom" &&
      !sameScopeSelection(orderedSelectedTools, initialTools.selected));
  const systemPromptDirty =
    instructionsSupported &&
    (systemPromptMode !== initialSystemPrompt.mode ||
      (systemPromptMode !== "standard" &&
        systemPromptText !== initialSystemPrompt.text));
  const systemPromptMissing =
    agentType === "openhands" &&
    instructionsSupported &&
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
      : toolConcurrency !== initialToolConcurrency ||
        systemPromptDirty ||
        behaviorDirty ||
        routerDirty);
  const credentialsDirty = acpCredentialForm.isDirty;
  const isAnyDirty =
    settingsDirty ||
    credentialsDirty ||
    (agentType === "openhands" && skillsDirty);
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

  const isAcp = agentType === "acp";
  // An OpenHands `default` shares its MCP servers, secrets and skills with the global pages.
  const sharesGlobalPages = isDefaultProfile && !isAcp;
  // Switching `default`'s agent type moves its MCP servers onto or off the
  // MCP page's switches.
  const sharedRef = React.useRef(sharesGlobalPages);
  React.useEffect(() => {
    if (sharedRef.current === sharesGlobalPages) return;
    sharedRef.current = sharesGlobalPages;
    const scope = sharesGlobalPages
      ? readProfileScope(enabledMcpServerRefs(settings))
      : initialMcpRefs;
    setMcpMode(scope.mode);
    setSelectedMcpServers(scope.selected);
  }, [sharesGlobalPages, settings, initialMcpRefs]);
  const commandTokens = parseCommand(commandText);
  const selectedPreset = detectPreset(commandText, acpProviders);
  const selectedProvider = getAcpProvider(selectedPreset);
  const isDefaultProviderCommand =
    !!selectedProvider &&
    commandTokens.join(" ") === selectedProvider.default_command.join(" ");
  const liveModels = useAcpModelDiscovery(
    isAcp && isDefaultProviderCommand ? selectedPreset : null,
    {
      secretRefs:
        secretRefsSupportedOnProfile && secretsMode === "custom"
          ? selectedSecrets
          : null,
    },
  );

  if (isLoading) return null;

  const modelSuggestions = buildModelSuggestions(
    liveModels.models,
    isCustomAcpModel ? "" : acpModel,
  );
  const agentDefaultModelLabel = labelForAcpModel(
    liveModels.defaultModelId,
    liveModels.models,
  );
  // Without a preferred model the agent picks its own, so name that choice.
  const offersAgentDefault =
    !!selectedProvider && !getAcpPreferredDefaultModel(selectedPreset);
  const showModelSelector = offersAgentDefault || modelSuggestions.length > 0;
  const modelListHint = (() => {
    if (!selectedProvider || liveModels.isDiscovering) return null;
    if (liveModels.discovery?.error?.code === "ACPAuthRequired") return null;
    if (liveModels.source === "remembered") {
      return {
        testId: "agent-model-list-remembered",
        key: I18nKey.SETTINGS$AGENT_MODEL_LIST_REMEMBERED,
      };
    }
    if (liveModels.source === "none") {
      return {
        testId: "agent-model-list-after-first-conversation",
        key: I18nKey.SETTINGS$AGENT_MODEL_LIST_AFTER_FIRST_CONVERSATION,
      };
    }
    return null;
  })();
  const selectedModelIsSuggestion = modelSuggestions.some(
    ({ id }) => id === acpModel.trim(),
  );
  let selectedModelKey = acpModel;
  if (offersAgentDefault && !isCustomAcpModel && !acpModel.trim()) {
    selectedModelKey = ACP_AGENT_DEFAULT_MODEL_KEY;
  } else if (isCustomAcpModel || !selectedModelIsSuggestion) {
    selectedModelKey = ACP_CUSTOM_MODEL_KEY;
  }
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
      skillEnablement: skillsToSave,
      toolsMode,
      selectedTools: orderedSelectedTools,
      toolParams: initialTools.params,
      toolCatalogLoaded,
      router: routerDirty ? router : undefined,
      systemPromptMode,
      systemPromptText,
      systemPromptEditable: instructionsSupported,
      condenser: behaviorToSave("condenser"),
      verification: behaviorToSave("verification"),
    });

  const isSaving = acpCredentialForm.isSaving;

  return (
    <div data-testid="agent-settings-screen" className="flex flex-col gap-4">
      <div
        data-testid="agent-profile-required-section"
        className="flex flex-col gap-5 rounded-xl border border-border bg-surface-raised p-5"
      >
        {profileNameField}
        {isAcp ? null : llmPicker}

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
                className="bg-tertiary border border-border-subtle rounded-sm p-2 text-sm font-mono text-contrast placeholder:text-tertiary-alt min-h-15 resize-y focus:outline-none focus:border-contrast"
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
                  const nextPreset = detectPreset(
                    nextCommandText,
                    acpProviders,
                  );
                  if (nextPreset !== prevPreset) {
                    setAcpModel(getAcpPreferredDefaultModel(nextPreset) ?? "");
                    setIsCustomAcpModel(false);
                    if (secretsMode === "custom")
                      addProviderSecrets(nextPreset);
                  }
                  setCommandText(nextCommandText);
                }}
              />
              <Typography.Text className="text-xs text-tertiary-alt">
                {t(I18nKey.SETTINGS$AGENT_COMMAND_HINT)}
              </Typography.Text>
            </div>
          </>
        )}

        {isAcp && selectedPreset !== ACP_CUSTOM_PRESET_KEY && (
          <>
            <hr className="border-border-subtle" />
            <AcpCredentialsSection
              form={acpCredentialForm}
              providerKey={selectedPreset}
              loginRejected={
                liveModels.discovery?.error?.code === "ACPAuthRequired"
              }
            />
            <hr className="border-border-subtle" />
          </>
        )}

        {isAcp && (
          <div className="flex flex-col gap-1.5">
            {showModelSelector && (
              <SettingsDropdownInput
                testId="agent-model-selector"
                name="agent-model"
                label={t(I18nKey.SETTINGS$AGENT_MODEL)}
                items={[
                  ...(offersAgentDefault
                    ? [
                        {
                          key: ACP_AGENT_DEFAULT_MODEL_KEY,
                          label: agentDefaultModelLabel
                            ? t(I18nKey.SETTINGS$AGENT_MODEL_AGENT_DEFAULT_IS, {
                                model: agentDefaultModelLabel,
                              })
                            : t(I18nKey.SETTINGS$AGENT_MODEL_AGENT_DEFAULT),
                        },
                      ]
                    : []),
                  ...modelSuggestions.map((model) => ({
                    key: model.id,
                    label:
                      !offersAgentDefault &&
                      model.id === liveModels.defaultModelId
                        ? t(I18nKey.SETTINGS$AGENT_MODEL_IS_AGENT_DEFAULT, {
                            model: model.label,
                          })
                        : model.label,
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
                  } else if (modelKey === ACP_AGENT_DEFAULT_MODEL_KEY) {
                    setIsCustomAcpModel(false);
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
                  showModelSelector
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
            {liveModels.isDiscovering && selectedProvider && (
              <Typography.Text
                testId="agent-model-discovering"
                className="text-xs text-tertiary-alt"
              >
                {t(I18nKey.SETTINGS$AGENT_MODEL_DISCOVERING, {
                  agent: selectedProvider.display_name,
                })}
              </Typography.Text>
            )}
            {liveModels.discovery?.error?.code === "ACPAuthRequired" &&
              selectedProvider && (
                <Typography.Text
                  testId="agent-model-discovery-needs-auth"
                  className="text-xs text-tertiary-alt"
                >
                  {t(I18nKey.SETTINGS$AGENT_MODEL_DISCOVERY_NEEDS_AUTH, {
                    agent: selectedProvider.display_name,
                  })}
                </Typography.Text>
              )}
            {modelListHint && selectedProvider && (
              <Typography.Text
                testId={modelListHint.testId}
                className="text-xs text-tertiary-alt"
              >
                {t(modelListHint.key, { agent: selectedProvider.display_name })}
              </Typography.Text>
            )}
            <Typography.Text className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$AGENT_MODEL_HINT)}
            </Typography.Text>
          </div>
        )}
      </div>

      {secretRefsSupportedOnProfile && (
        <AgentFormSection
          testId="agent-secrets-section"
          title={t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS)}
        >
          {sharesGlobalPages ? (
            <div className="flex flex-col gap-2.5 pt-4">
              <Typography.Text className="text-sm">
                {t(I18nKey.SETTINGS$AGENT_PROFILE_SECRETS)}
              </Typography.Text>
              <Typography.Text
                testId="agent-settings-secrets-default-note"
                className="text-xs text-tertiary-alt"
              >
                {t(I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_SECRETS_NOTE)}
              </Typography.Text>
              {secretCatalog.length > 0 && (
                <ProfileScopeList
                  testId="agent-settings-secret"
                  items={secretCatalog}
                  selected={secretCatalog.map(({ name }) => name)}
                  onDelete={setSecretToDelete}
                  deletableNames={savedSecrets?.map(({ name }) => name)}
                  isDeleteDisabled={isSaving || deleteSecret.isPending}
                />
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-2.5 pt-4">
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
                  onDelete={setSecretToDelete}
                  deletableNames={savedSecrets?.map(({ name }) => name)}
                  isDeleteDisabled={isSaving || deleteSecret.isPending}
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
          )}
          {addingSecret ? (
            <SecretForm
              mode="add"
              selectedSecret={null}
              keepOpenOnError
              onCancel={() => setAddingSecret(false)}
              onSaved={({ name }) => {
                setDeletedSecrets((deleted) =>
                  deleted.filter((entry) => entry !== name),
                );
                if (secretsMode === "custom") {
                  setSelectedSecrets((selected) => [
                    ...new Set([...selected, name]),
                  ]);
                }
              }}
            />
          ) : (
            <BrandButton
              type="button"
              variant="secondary"
              testId="agent-add-secret"
              onClick={() => setAddingSecret(true)}
            >
              {t(I18nKey.SECRETS$ADD_A_SECRET)}
            </BrandButton>
          )}
        </AgentFormSection>
      )}
      {secretToDelete && (
        <ConfirmationModal
          text={`${t(I18nKey.SECRETS$CONFIRM_DELETE_KEY)} ${t(I18nKey.SETTINGS$SHARED_RESOURCE_DELETE_WARNING)}`}
          onCancel={() => setSecretToDelete(null)}
          onConfirm={() => void handleConfirmDeleteSecret()}
          confirmText={t(I18nKey.BUTTON$DELETE)}
          isDestructive
          isConfirming={deleteSecret.isPending}
        />
      )}

      <AgentFormSection
        testId="agent-mcp-section"
        title={t(I18nKey.SETTINGS$AGENT_PROFILE_MCP)}
      >
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
                setSelectedMcpServers(
                  configuredMcpNames.map(({ name }) => name),
                );
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
          <Typography.Text
            testId={
              sharesGlobalPages ? "agent-settings-mcp-default-note" : undefined
            }
            className="text-xs text-tertiary-alt"
          >
            {t(
              sharesGlobalPages
                ? I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_MCP_NOTE
                : mcpMode === "custom"
                  ? I18nKey.SETTINGS$AGENT_PROFILE_MCP_CHOOSE_HINT
                  : I18nKey.SETTINGS$AGENT_PROFILE_MCP_ALL_HINT,
            )}
          </Typography.Text>
          <AgentMcpSetup
            servers={flattenMcpConfig(
              settings?.mcp_config ??
                parseMcpConfig(settings?.agent_settings?.mcp_config),
            )}
            onServerDeleted={(serverId) =>
              setSelectedMcpServers((selected) =>
                selected.filter((name) => name !== serverId),
              )
            }
          />
        </div>
      </AgentFormSection>

      {agentType === "openhands" && skillsSupported ? (
        <AgentFormSection
          testId="agent-settings-skills-section"
          title={t(I18nKey.SETTINGS$AGENT_PROFILE_SKILLS)}
        >
          <div className="flex flex-col gap-2.5">
            {skillCatalog && skillCatalog.length > SKILL_SEARCH_MIN_ITEMS ? (
              <SettingsInput
                testId="agent-settings-skills-search"
                label=""
                type="search"
                className="w-full"
                value={skillQuery}
                placeholder={t(I18nKey.SETTINGS$SKILLS_SEARCH_PLACEHOLDER)}
                onChange={setSkillQuery}
              />
            ) : null}
            {visibleSkills && currentSkills ? (
              <div className="max-h-80 overflow-y-auto pr-1">
                <ProfileScopeList
                  testId="agent-settings-skills"
                  items={visibleSkills.map((skill) => ({
                    name: skill.name,
                    description: t(
                      skill.origin === "public"
                        ? I18nKey.SETTINGS$SKILL_ORIGIN_PUBLIC
                        : I18nKey.SETTINGS$SKILL_ORIGIN_USER,
                    ),
                  }))}
                  selected={enabledSkillNames}
                  isDisabled={isSaving}
                  onToggle={toggleProfileSkill}
                />
              </div>
            ) : null}
            {skillCatalogFailed ? (
              <div
                className="flex items-center gap-2"
                data-testid="agent-settings-skills-load-failed"
              >
                <Typography.Text className="text-xs text-feedback-error">
                  {t(I18nKey.SETTINGS$AGENT_PROFILE_SKILLS_LOAD_FAILED)}
                </Typography.Text>
                <BrandButton
                  testId="agent-settings-skills-retry"
                  type="button"
                  variant="secondary"
                  onClick={() => refetchSkillCatalog()}
                >
                  {t(I18nKey.SETTINGS$AGENT_PROFILE_TOOLS_RETRY)}
                </BrandButton>
              </div>
            ) : null}
            <Typography.Text
              testId={
                sharesGlobalPages
                  ? "agent-settings-skills-default-note"
                  : undefined
              }
              className="text-xs text-tertiary-alt"
            >
              {t(
                sharesGlobalPages
                  ? I18nKey.SETTINGS$AGENT_PROFILE_DEFAULT_SKILLS_NOTE
                  : I18nKey.SETTINGS$AGENT_PROFILE_SKILLS_HINT,
              )}
            </Typography.Text>
            <BrandButton
              type="button"
              variant="secondary"
              testId="agent-add-skill-help"
              onClick={() => setShowAddSkillHelp(true)}
            >
              {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_TITLE)}
            </BrandButton>
            {showAddSkillHelp && (
              <AddSkillModal onClose={() => setShowAddSkillHelp(false)} />
            )}
          </div>
        </AgentFormSection>
      ) : null}

      <AgentFormSection
        testId="agent-behavior-section"
        title={t(I18nKey.SETTINGS$AGENT_PROFILE_BEHAVIOR)}
      >
        {!isAcp && instructionsSupported ? (
          <div className="flex flex-col gap-2.5">
            <Typography.Text className="text-sm">
              {t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT)}
            </Typography.Text>
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
                  label: t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_APPEND),
                },
                {
                  key: "custom",
                  label: t(I18nKey.SETTINGS$AGENT_PROFILE_SYSTEM_PROMPT_CUSTOM),
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
          </div>
        ) : null}

        {!isAcp && routerSupported ? (
          <div
            className="flex flex-col gap-2.5"
            data-testid="agent-settings-router"
          >
            <Typography.Text className="text-sm">
              {t(I18nKey.SETTINGS$NAV_META_LLM)}
            </Typography.Text>
            <SettingsDropdownInput
              testId="agent-settings-router-select"
              name="agent-router"
              label=""
              items={[
                {
                  key: ROUTER_OFF_KEY,
                  label: t(I18nKey.SETTINGS$AGENT_PROFILE_ROUTER_OFF),
                },
                ...(initialRouter === ROUTER_FIRST_KEY
                  ? [
                      {
                        key: ROUTER_FIRST_KEY,
                        label: t(I18nKey.SETTINGS$AGENT_PROFILE_ROUTER_FIRST),
                      },
                    ]
                  : []),
                ...[
                  ...routerNames,
                  ...(routerDangling && router ? [router] : []),
                ].map((name) => ({ key: name, label: name })),
              ]}
              selectedKey={router ?? ROUTER_OFF_KEY}
              isDisabled={isSaving}
              onSelectionChange={(key) => {
                if (key === null || key === undefined) return;
                setRouter(key === ROUTER_OFF_KEY ? null : String(key));
              }}
            />
            {routerDangling ? (
              <Typography.Text
                testId="agent-settings-router-dangling"
                className="text-xs text-danger"
              >
                {t(I18nKey.SETTINGS$AGENT_PROFILE_ROUTER_DANGLING, {
                  name: router,
                })}
              </Typography.Text>
            ) : null}
            <Typography.Text className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$AGENT_PROFILE_ROUTER_HINT)}
            </Typography.Text>
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
        {!isAcp && toolsSupported ? (
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
                <Typography.Text className="text-xs text-feedback-error">
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

        {!isAcp
          ? PROFILE_BEHAVIOR_BLOCKS.map(({ section, titleKey }) => (
              <div
                key={section}
                className="flex flex-col gap-2.5"
                data-testid={`agent-settings-${section}`}
              >
                <Typography.Text className="text-sm">
                  {t(titleKey)}
                </Typography.Text>
                <SdkSectionPage
                  settingsSources={[
                    {
                      settingsSource: "agent_settings",
                      sectionKeys: [section],
                      excludeKeys: GLOBAL_BEHAVIOR_FIELD_KEYS,
                    },
                  ]}
                  embedded
                  hideSaveButton
                  markInitialOverridesDirty={false}
                  initialValueOverrides={behaviorInitialValues[section]}
                  getInitialView={behaviorInitialView[section]}
                  onSaveControlChange={behaviorControlHandlers[section]}
                  testId={`agent-settings-${section}-fields`}
                />
              </div>
            ))
          : null}
      </AgentFormSection>
    </div>
  );
}

/**
 * Legacy `/settings/agent` route. Settings → Agent is now the Agent Profile
 * library (`/agents`), whose editor reuses the named
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
  return <Navigate to="/agents" replace />;
}
