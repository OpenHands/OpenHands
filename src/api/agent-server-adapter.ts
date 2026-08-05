import { ACP_SETTINGS_KEYS } from "@openhands/typescript-client";
import { ServerClient } from "@openhands/typescript-client/clients";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { ExecutionStatus } from "#/types/agent-server/core";
import { AgentKind, Settings, SettingsValue } from "#/types/settings";
import {
  getAcpPreferredDefaultModel,
  getAcpProvider,
  resolveEffectiveAcpModel,
} from "#/constants/acp-providers";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import {
  getCachedAgentServerInfo,
  isAgentServerToolAvailable,
} from "./agent-server-compatibility";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { buildAuthHeaders } from "./backend-registry/auth";
import {
  GetHooksResponse,
  PluginSpec,
  AppConversation,
  AppConversationPage,
  SandboxStatus,
} from "./conversation-service/agent-server-conversation-service.types";
import SettingsService from "./settings-service/settings-service.api";
import { getStoredConversationMetadata } from "./conversation-metadata-store";
import LLMSubscriptionService from "./llm-subscription-service";
import {
  LLM_AUTH_TYPE_SUBSCRIPTION,
  OPENAI_SUBSCRIPTION_VENDOR,
  isSubscriptionLlmConfig,
} from "#/constants/llm-subscription";
import {
  CANVAS_UI_CLIENT_TOOL,
  CANVAS_UI_CLIENT_TOOL_NAME,
  LEGACY_CANVAS_UI_TOOL_NAME,
  type ClientToolSpec,
} from "./canvas-ui-client-tool";
import {
  buildPlanPath,
  LOCAL_PLANNER_PARENT_TAG_KEY,
  PLAN_STRUCTURE_TEXT,
  PLANNING_AGENT_INSTRUCTION,
  PLANNING_FILE_EDITOR_TOOL_NAME,
  PLANNING_SYSTEM_PROMPT_FILENAME,
} from "#/utils/plan-file";

export interface DirectConversationInfo {
  id: string;
  title?: string | null;
  created_at: string;
  updated_at: string;
  execution_status?: string | null;
  /** Cloud-only sandbox lifecycle state. Omitted / null for local agent-server conversations. */
  sandbox_status?: string | null;
  metrics?: {
    accumulated_cost?: number | null;
    max_budget_per_task?: number | null;
    accumulated_token_usage?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      cache_read_tokens?: number;
      cache_write_tokens?: number;
      context_window?: number;
      per_turn_token?: number;
    } | null;
  } | null;
  agent?: {
    /**
     * Pydantic discriminator from the SDK union: ``"ACPAgent"`` for ACP CLI
     * subprocesses (model lives on the subprocess via ``acp_model``),
     * ``"Agent"`` for direct litellm. Read by {@link toAppConversation}.
     */
    kind?: string | null;
    acp_model?: string | null;
    /**
     * ACP CLI identity (``claude-code`` / ``codex`` / ``gemini-cli``) from the
     * SDK's ``ACPAgent.acp_server`` (#3692). Preferred fallback when the
     * ``acpserver`` tag is absent — e.g. a profile launch doesn't stamp the tag
     * client-side and the server may not repopulate it. Read by {@link toAppConversation}.
     */
    acp_server?: string | null;
    llm?: {
      model?: string | null;
    } | null;
  } | null;
  current_model_id?: string | null;
  current_model_name?: string | null;
  workspace?: {
    working_dir?: string | null;
  } | null;
  /**
   * Arbitrary string-keyed conversation tags surfaced by the agent-server
   * (see ``ConversationInfo.tags``). Canvas only consumes one key today —
   * ``ACP_SERVER_TAG_KEY`` ("acpserver") — but the field is typed as a
   * generic record so future readers don't need another wire-shape change.
   * Keys are constrained to ``^[a-z0-9]+$`` by the agent-server validator;
   * values are opaque strings.
   */
  tags?: Record<string, string> | null;
  launched_agent_profile?: {
    agent_profile_id: string;
    revision: number;
  } | null;
  /**
   * Server-owned, derived from the catalog's ``parent_conversation_id`` link
   * (agent-server >= 1.37.1, SDK #4188) — a conversation started with a
   * parent is discoverable here on any browser, which is what makes the
   * local planner's relationship server state rather than a browser-local
   * hint. Absent on older agent-servers and on the cloud wire shape.
   */
  sub_conversation_ids?: string[] | null;
}

const DEFAULT_TOOL_NAMES = ["terminal", "file_editor", "task_tracker"];
const BROWSER_TOOL_SET_NAME = "browser_tool_set";
const TASK_TOOL_SET_NAME = "task_tool_set";
// Falls back to the same default the code agent uses when the user has not
// configured `conversation_settings.max_iterations` (see buildConfiguredConversationSettings).
const DEFAULT_MAX_ITERATIONS = 500;

function resolveMaxIterations(value: unknown): number {
  return typeof value === "number" ? value : DEFAULT_MAX_ITERATIONS;
}

function browserToolsEnabled() {
  return import.meta.env.VITE_ENABLE_BROWSER_TOOLS !== "false";
}

/**
 * Shape of the runtime services info served by Agent Canvas backends in
 * `/server_info.runtime_services`. All URLs are written from the agent's point of
 * view, not the browser's. The block is rendered into the agent's system prompt
 * via `AgentContext.system_message_suffix` so the agent knows what's reachable
 * from inside its sandbox without having to probe.
 */
export interface RuntimeServicesInfo {
  mode?: string;
  agent_host_alias?: string;
  services?: {
    agent_server?: { description?: string; url_from_agent?: string };
    ingress?: { description?: string; url_from_agent?: string };
    frontend?: {
      kind?: "vite" | "static";
      description?: string;
      url_from_agent?: string;
    };
    // `vite` is the legacy key name for the frontend entry, accepted for
    // one release while older dev-stack launchers may still emit it.
    vite?: { description?: string; url_from_agent?: string };
    automation?: {
      description?: string;
      url_from_agent?: string;
      api_prefix?: string;
      docs_url?: string;
      openapi_url?: string;
      auth_env_var?: string;
    };
  };
}

export function parseRuntimeServicesInfo(
  value: unknown,
): RuntimeServicesInfo | null {
  if (typeof value === "string") {
    const raw = value.trim();
    if (!raw) return null;
    try {
      return parseRuntimeServicesInfo(JSON.parse(raw));
    } catch {
      return null;
    }
  }

  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const parsed = value as RuntimeServicesInfo;
  if (!parsed.services || typeof parsed.services !== "object") return null;
  return parsed;
}

export async function fetchBackendRuntimeServicesInfo(): Promise<RuntimeServicesInfo | null> {
  let clientOptions: ReturnType<typeof getAgentServerClientOptions>;
  try {
    clientOptions = getAgentServerClientOptions({ timeout: 3000 });
  } catch {
    return null;
  }

  const cached = parseRuntimeServicesInfo(
    getCachedAgentServerInfo({ host: clientOptions.host })?.runtime_services,
  );
  if (cached) return cached;

  try {
    const serverInfo = await new ServerClient(clientOptions).getServerInfo();
    return parseRuntimeServicesInfo(
      (serverInfo as { runtime_services?: unknown }).runtime_services,
    );
  } catch {
    return null;
  }
}

/**
 * Return the deployment mode from the runtime services info, e.g. "docker",
 * "dev:automation", etc. Returns `null` when no runtime info is supplied.
 */
export function getDeploymentMode(
  runtimeServicesInfo?: RuntimeServicesInfo | null,
): string | null {
  return runtimeServicesInfo?.mode ?? null;
}

/**
 * Render the runtime services info into a markdown block suitable for
 * appending to the system prompt via `AgentContext.system_message_suffix`.
 *
 * Returns `undefined` when no runtime info is available, so callers can safely
 * omit the field when the selected backend does not advertise runtime services.
 */
export function buildRuntimeServicesSystemSuffix(
  runtimeServicesInfo?: RuntimeServicesInfo | null,
): string | undefined {
  const info = parseRuntimeServicesInfo(runtimeServicesInfo);
  if (!info?.services) return undefined;

  const lines: string[] = [];
  lines.push("<RUNTIME_SERVICES>");
  if (info.mode) {
    lines.push(
      `You are running inside an agent-canvas dev stack started in '${info.mode}' mode.`,
    );
  } else {
    lines.push("You are running inside an agent-canvas dev stack.");
  }
  lines.push(
    "The following services are reachable from your sandbox. URLs are written",
    "from your point of view (i.e., as you should curl/fetch them).",
    "",
  );

  const { agent_server, ingress, automation } = info.services;
  const { frontend } = info.services;

  if (agent_server?.url_from_agent) {
    lines.push(
      `* Agent Server (you): ${agent_server.url_from_agent}`,
      `    ${agent_server.description ?? "The agent-server hosting your tool calls."}`,
    );
  }
  if (ingress?.url_from_agent) {
    lines.push(
      `* Ingress: ${ingress.url_from_agent}`,
      `    ${ingress.description ?? "Unified entry point for browser-facing traffic."}`,
    );
  }
  if (frontend?.url_from_agent) {
    lines.push(
      `* Frontend: ${frontend.url_from_agent}`,
      `    ${frontend.description ?? "Frontend dev server."}`,
    );
  }
  if (automation?.url_from_agent) {
    lines.push(
      `* Automation backend: ${automation.url_from_agent}`,
      `    ${automation.description ?? "OpenHands Automations service."}`,
    );
    if (automation.docs_url) {
      lines.push(`    Docs:    ${automation.docs_url}`);
    }
    if (automation.openapi_url) {
      lines.push(`    OpenAPI: ${automation.openapi_url}`);
    }
    if (automation.auth_env_var) {
      // X-Session-API-Key is the local convention shared by the agent-server
      // and automation backend (see openhands-automation auth.py).
      lines.push(
        `    Auth:    header 'X-Session-API-Key: $${automation.auth_env_var}'`,
      );
    }
  } else {
    lines.push(
      "* Automation backend: not running in this dev mode (skip /api/automation calls).",
    );
  }

  // Anchor the "don't guess" warning to the actual agent-server URL for
  // this stack instead of a hardcoded port. The agent-server listens on
  // different ports across dev modes, and baking the wrong port into the
  // system prompt is exactly the kind of confusion this block is meant to
  // prevent.
  const agentServerUrl = agent_server?.url_from_agent;
  lines.push(
    "",
    "Trust this block over guessing: do not assume any other URLs are running.",
  );
  if (agentServerUrl) {
    lines.push(
      `In particular, ${agentServerUrl} inside your sandbox is the Agent Server`,
      "you are running inside of — NOT the automation backend.",
    );
  }
  lines.push("</RUNTIME_SERVICES>");

  return lines.join("\n");
}

export function toConversationUrl(conversationId: string): string {
  // Local-format conversation URL — points at whichever local agent-server
  // is actually serving the conversation (the bundled one when the active
  // selection is cloud).
  const { host } = getAgentServerClientOptions();
  return `${host}/api/conversations/${conversationId}`;
}

// TODO(i18n): extract "Conversation" once we add CONVERSATION$DEFAULT_TITLE
// with `{{shortId}}` interpolation. Kept as a literal for now to keep the
// fallback inside this pure adapter rather than fanning out to display sites.
export function getDefaultConversationTitle(conversationId: string): string {
  return `Conversation ${conversationId.slice(0, 5)}`;
}

export function toAppConversation(
  info: DirectConversationInfo,
): AppConversation {
  const metadata = getStoredConversationMetadata(info.id);
  // ACPAgent conversations carry a sentinel ``llm`` on older SDKs. Prefer the
  // runtime model fields when available, then the configured ``acp_model`` that
  // Canvas saves for built-in providers. ``agent_kind`` still gates model
  // switching, so surfacing this string is display-only.
  const isAcp = info.agent?.kind === "ACPAgent";
  // Only surface ``acp_server`` for ACP conversations even if the wire
  // payload accidentally carries an ``acpserver`` tag on an OpenHands
  // conversation — the chip is identity info for the ACP CLI subprocess,
  // and showing it on a non-ACP conversation would be a lie. Fall back to the
  // agent's own ``acp_server`` (#3692) when the tag is missing — a profile
  // launch doesn't stamp the tag, so tag-only reads would drop the ACP model
  // picker and degrade the chip to a generic "ACP" (#1571).
  const acpServer = isAcp
    ? (info.tags?.[ACP_SERVER_TAG_KEY] ?? info.agent?.acp_server ?? null)
    : null;
  return {
    id: info.id,
    created_by_user_id: null,
    selected_repository: metadata?.selected_repository ?? null,
    selected_branch: metadata?.selected_branch ?? null,
    git_provider: metadata?.git_provider ?? null,
    selected_workspace: metadata?.selected_workspace ?? null,
    active_profile: metadata?.active_profile ?? null,
    title: info.title?.trim()
      ? info.title
      : getDefaultConversationTitle(info.id),
    trigger: null,
    pr_number: [],
    agent_kind: isAcp ? "acp" : "openhands",
    acp_server: acpServer,
    tags: info.tags ?? null,
    launched_agent_profile: info.launched_agent_profile ?? null,
    // Chip path: omit ``providerDefault`` so that when no concrete model
    // resolves, the chip falls back to the provider display name in
    // ConversationCardFooter rather than a registry default the session may
    // not actually be running.
    llm_model: isAcp
      ? resolveEffectiveAcpModel({
          runtimeName: info.current_model_name,
          runtimeId: info.current_model_id,
          configured: info.agent?.acp_model,
          sdkLlm: info.agent?.llm?.model,
        })
      : (info.agent?.llm?.model ?? DEFAULT_SETTINGS.llm_model),
    metrics: info.metrics
      ? {
          accumulated_cost: info.metrics.accumulated_cost ?? null,
          max_budget_per_task: info.metrics.max_budget_per_task ?? null,
          accumulated_token_usage: info.metrics.accumulated_token_usage
            ? {
                prompt_tokens:
                  info.metrics.accumulated_token_usage.prompt_tokens ?? 0,
                completion_tokens:
                  info.metrics.accumulated_token_usage.completion_tokens ?? 0,
                cache_read_tokens:
                  info.metrics.accumulated_token_usage.cache_read_tokens ?? 0,
                cache_write_tokens:
                  info.metrics.accumulated_token_usage.cache_write_tokens ?? 0,
                context_window:
                  info.metrics.accumulated_token_usage.context_window ?? 0,
                per_turn_token:
                  info.metrics.accumulated_token_usage.per_turn_token ?? 0,
              }
            : null,
        }
      : null,
    created_at: info.created_at,
    updated_at: info.updated_at,
    execution_status:
      (info.execution_status as AppConversation["execution_status"]) ??
      ExecutionStatus.IDLE,
    sandbox_status: (info.sandbox_status as SandboxStatus | null) ?? null,
    conversation_url: toConversationUrl(info.id),
    session_api_key: getAgentServerClientOptions().apiKey ?? null,
    sandbox_id: null,
    workspace: {
      working_dir: info.workspace?.working_dir ?? getAgentServerWorkingDir(),
    },
    public: false,
    sub_conversation_ids: info.sub_conversation_ids ?? [],
  };
}

export function toConversationPage(data: {
  items: DirectConversationInfo[];
  next_page_id?: string | null;
}): AppConversationPage {
  return {
    items: data.items
      .filter((item) => !item.tags?.[LOCAL_PLANNER_PARENT_TAG_KEY])
      .map(toAppConversation),
    next_page_id: data.next_page_id ?? null,
  };
}

type SettingsRecord = Record<string, unknown>;

interface AgentToolSpec {
  name: string;
  params: SettingsRecord;
}

type AgentSettingsPayload = SettingsRecord & {
  llm?: SettingsRecord;
  agent_context: SettingsRecord;
  tools?: AgentToolSpec[];
};

interface LocalWorkspacePayload {
  kind: "LocalWorkspace";
  working_dir: string;
}

interface InitialMessagePayload {
  role: "user";
  content: Array<{ type: "text"; text: string }>;
  run: true;
}

type ConversationSettingsPayload = SettingsRecord & {
  workspace: LocalWorkspacePayload;
  initial_message?: InitialMessagePayload;
};

export const ACP_SERVER_TAG_KEY = "acpserver";

/**
 * Conversation tag keys Canvas itself stamps/consumes for internal routing.
 * They are already surfaced through dedicated UI (the ACP provider chip,
 * the hidden-from-list planner filter) or not meant to be user-facing at
 * all, so the generic tag-chip display filters them out.
 */
export const RESERVED_CONVERSATION_TAG_KEYS: ReadonlySet<string> = new Set([
  ACP_SERVER_TAG_KEY,
  LOCAL_PLANNER_PARENT_TAG_KEY,
]);

/**
 * User-facing subset of a conversation's server-side tags: everything except
 * {@link RESERVED_CONVERSATION_TAG_KEYS}, as stable ``[key, value]`` entries
 * sorted by key so chip order doesn't shuffle between refetches.
 */
export function getDisplayConversationTags(
  tags: Record<string, string> | null | undefined,
): Array<[string, string]> {
  if (!tags) {
    return [];
  }
  return Object.entries(tags)
    .filter(([key]) => !RESERVED_CONVERSATION_TAG_KEYS.has(key))
    .sort(([a], [b]) => a.localeCompare(b));
}

const FERNET_TOKEN_PREFIX = "gAAAAA";

const CONVERSATION_SETTINGS_METADATA_KEYS = new Set([
  "schema_version",
  "agent_settings",
  "workspace",
  "conversation_id",
  "initial_message",
  "plugins",
]);

function toRecord(value: unknown): SettingsRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  return structuredClone(value as SettingsRecord);
}

function normalizeSecretString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function buildNormalizedLlmSettings(value: unknown): SettingsRecord {
  const llm = toRecord(value);

  llm.model =
    typeof llm.model === "string" && llm.model.trim().length > 0
      ? llm.model
      : DEFAULT_SETTINGS.llm_model;

  const apiKey = normalizeSecretString(llm.api_key);
  if (apiKey) {
    llm.api_key = apiKey;
  } else {
    delete llm.api_key;
  }

  const baseUrl = normalizeSecretString(llm.base_url);
  if (baseUrl) {
    llm.base_url = baseUrl;
  } else {
    delete llm.base_url;
  }

  if (isSubscriptionLlmConfig(llm)) {
    llm.auth_type = LLM_AUTH_TYPE_SUBSCRIPTION;
    llm.subscription_vendor = OPENAI_SUBSCRIPTION_VENDOR;
    delete llm.api_key;
    delete llm.base_url;
  } else {
    delete llm.auth_type;
    delete llm.subscription_vendor;
  }

  return llm;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function hasEncryptedString(value: unknown): boolean {
  if (typeof value === "string") {
    return value.startsWith(FERNET_TOKEN_PREFIX);
  }
  if (Array.isArray(value)) {
    return value.some(hasEncryptedString);
  }
  if (isPlainRecord(value)) {
    return Object.values(value).some(hasEncryptedString);
  }
  return false;
}

function hasEncryptedMcpSecrets(mcpConfig: unknown): boolean {
  if (!isPlainRecord(mcpConfig)) {
    return false;
  }

  return Object.values(mcpConfig).some(hasEncryptedString);
}

function getConversationConfirmationPolicy(
  conversationSettings: SettingsRecord,
) {
  if (conversationSettings.confirmation_mode !== true) {
    return { kind: "NeverConfirm" };
  }

  if (conversationSettings.security_analyzer === "llm") {
    return { kind: "ConfirmRisky", threshold: "HIGH", confirm_unknown: true };
  }

  return { kind: "AlwaysConfirm" };
}

function getConversationSecurityAnalyzer(conversationSettings: SettingsRecord) {
  switch (conversationSettings.security_analyzer) {
    case "llm":
      return { kind: "LLMSecurityAnalyzer" };
    case "pattern":
      return { kind: "PatternSecurityAnalyzer" };
    case "policy_rail":
      return { kind: "PolicyRailSecurityAnalyzer" };
    default:
      return undefined;
  }
}

function isToolRecord(
  value: unknown,
): value is { name: string; params?: unknown } {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { name?: unknown }).name === "string"
  );
}

function shouldIncludeTool(name: string, agentSettings: SettingsRecord) {
  if (name === BROWSER_TOOL_SET_NAME) {
    return browserToolsEnabled() && isAgentServerToolAvailable(name);
  }

  if (name === TASK_TOOL_SET_NAME) {
    return (
      agentSettings.enable_sub_agents === true &&
      isAgentServerToolAvailable(name)
    );
  }

  return true;
}

function getAgentTools(agentSettings: SettingsRecord): AgentToolSpec[] {
  const tools = new Map<string, AgentToolSpec>();

  for (const name of DEFAULT_TOOL_NAMES) {
    if (shouldIncludeTool(name, agentSettings)) {
      tools.set(name, { name, params: {} });
    }
  }

  for (const name of [BROWSER_TOOL_SET_NAME, TASK_TOOL_SET_NAME]) {
    if (shouldIncludeTool(name, agentSettings)) {
      tools.set(name, { name, params: {} });
    }
  }

  const configuredTools = agentSettings.tools;
  if (
    Array.isArray(configuredTools) &&
    configuredTools.every((tool) => isToolRecord(tool))
  ) {
    for (const tool of configuredTools) {
      if (shouldIncludeTool(tool.name, agentSettings)) {
        tools.set(tool.name, {
          name: tool.name,
          params: toRecord(tool.params),
        });
      }
    }
  }

  return Array.from(tools.values());
}

function buildInitialMessage(
  query?: string,
  conversationInstructions?: string,
): InitialMessagePayload | null {
  const parts = [query?.trim(), conversationInstructions?.trim()].filter(
    Boolean,
  );
  if (parts.length === 0) {
    return null;
  }

  return {
    role: "user",
    content: [{ type: "text", text: parts.join("\n\n") }],
    run: true,
  };
}

/**
 * Shape of a bundled skill entry passed to the agent-server SDK via
 * `agent_context.skills`. Mirrors the SDK's `Skill` model fields that
 * the server uses for trigger matching, activation, and system-prompt
 * injection.
 */
interface BundledSkill {
  name: string;
  content: string;
  trigger: { type: "keyword"; keywords: string[] } | null;
  source: string;
  description: string | null;
  is_agentskills_format: true;
  license?: string;
  compatibility?: string;
}

/**
 * Convert the bundled `SKILLS_CATALOG` entries into the SDK `Skill` JSON
 * shape so the agent-server can perform trigger matching, skill activation,
 * and system-prompt injection without cloning the extensions repo.
 *
 * The SDK discriminates triggers via `{ type: "keyword", keywords: [...] }`.
 * Skills with no triggers get `trigger: null` (always-active / on-demand).
 */
function buildBundledSkills(): BundledSkill[] {
  return SKILLS_CATALOG.map((entry) => {
    const trigger: BundledSkill["trigger"] =
      entry.triggers?.length > 0
        ? { type: "keyword", keywords: entry.triggers }
        : null;

    // Use the absolute path to the skill's SKILL.md so the Python
    // agent-server can resolve bundled resources (scripts/, references/).
    // Falls back to "public" in library builds where the path isn't known.
    const source = __EXTENSIONS_SKILLS_DIR__
      ? `${__EXTENSIONS_SKILLS_DIR__}/${entry.name}/SKILL.md`
      : "public";

    return {
      name: entry.name,
      content: entry.content,
      trigger,
      source,
      description: entry.description ?? null,
      is_agentskills_format: true as const,
      ...(entry.license ? { license: entry.license } : {}),
      ...(entry.compatibility ? { compatibility: entry.compatibility } : {}),
    };
  });
}

function buildAgentContext(
  agentSettings: SettingsRecord,
  runtimeServicesInfo?: RuntimeServicesInfo | null,
  disabledSkills: string[] = [],
): SettingsRecord {
  const runtimeServicesSuffix =
    buildRuntimeServicesSystemSuffix(runtimeServicesInfo);
  const existingContext = toRecord(agentSettings.agent_context);

  // Merge bundled public skills with any skills already present in the
  // agent context (e.g. user-defined skills set via the settings API).
  const existingSkills = Array.isArray(existingContext.skills)
    ? (existingContext.skills as SettingsRecord[])
    : [];
  const disabledSkillNames = new Set(disabledSkills);
  const mergedSkills = [...existingSkills, ...buildBundledSkills()].filter(
    (skill) =>
      typeof skill.name !== "string" || !disabledSkillNames.has(skill.name),
  );

  return {
    ...existingContext,
    // Public skills are bundled at build time from the @openhands/extensions
    // npm package and passed directly in agent_context.skills. Setting
    // load_public_skills to false tells the agent-server SDK to skip its own
    // extensions-repo clone — the frontend is the sole source of public
    // skills now.
    //
    // Migration: the former VITE_LOAD_PUBLIC_SKILLS env var was removed
    // because bundled skills have no clone latency. Users who previously set
    // VITE_LOAD_PUBLIC_SKILLS=false to avoid clone delays no longer need it.
    skills: mergedSkills,
    load_public_skills: false,
    load_user_skills: true,
    load_project_skills: true,
    ...(runtimeServicesSuffix
      ? { system_message_suffix: runtimeServicesSuffix }
      : {}),
  };
}

function isAcpAgent(settings: Settings): boolean {
  const agentSettings = toRecord(settings.agent_settings);
  return agentSettings.agent_kind === "acp";
}

function getAcpServerTag(settings: Settings): string | undefined {
  const agentSettings = toRecord(settings.agent_settings);
  const value = agentSettings.acp_server;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function resolveAcpCommand(agentSettings: SettingsRecord): unknown {
  const cmd = agentSettings.acp_command;
  const isEmpty = Array.isArray(cmd) && cmd.length === 0;
  const noCommand = cmd === undefined;
  if (!isEmpty && !noCommand) {
    return cmd;
  }

  const serverKey =
    typeof agentSettings.acp_server === "string"
      ? agentSettings.acp_server
      : undefined;
  const provider = getAcpProvider(serverKey);
  return provider ? [...provider.default_command] : cmd;
}

function buildConfiguredAcpAgentSettings(
  settings: Settings,
  runtimeServicesInfo?: RuntimeServicesInfo | null,
): AgentSettingsPayload {
  const agentSettings = toRecord(settings.agent_settings);
  const payload: AgentSettingsPayload = {
    agent_kind: "acp",
    agent_context: buildAgentContext(
      agentSettings,
      runtimeServicesInfo,
      settings.disabled_skills,
    ),
  };

  // TODO(#1019): set ``acp_isolate_data_dir: true`` here for a containerized
  // backend so concurrent same-provider conversations don't race on a shared
  // HOME. The SDK supports it (software-agent-sdk#3492), but the released
  // ``@openhands/typescript-client`` (1.24.3) doesn't surface it on
  // ``ACPAgentSettings`` yet, so sending it risks a validation error on older
  // servers. Cloud grouping isolation is separate (agent-canvas#1016).

  for (const key of ACP_SETTINGS_KEYS) {
    // ``acp_model`` is resolved separately below so a saved ``null`` still
    // falls back to the provider's default rather than being dropped.
    if (key === "acp_model") continue;
    // ``acp_env`` is deprecated — provider creds now route via ``request.secrets``.
    if (key === "acp_env") continue;
    const value =
      key === "acp_command"
        ? resolveAcpCommand(agentSettings)
        : agentSettings[key];
    if (value !== undefined && value !== null) {
      payload[key] = value;
    }
  }

  // ``mcp_config`` is a shared field (not in ACP_SETTINGS_KEYS): forward it
  // so the ACP subprocess connects to the configured MCP servers at session
  // creation. Only include it when it actually carries servers — an empty or
  // malformed value is dropped rather than sending ``mcp_config: {}``.
  const mcpConfig = toRecord(agentSettings.mcp_config);
  if (Object.keys(mcpConfig).length > 0) {
    payload.mcp_config = mcpConfig;
  }

  // Saved settings may carry ``acp_model: null`` (existing users predating
  // the default-model registry, or saved fields the agent-server stripped).
  // Fall back to the *preferred* default (Vertex-safe for Gemini) so the
  // conversation starts with whatever the Settings → Agent UI shows — without
  // that, the form's displayed default would silently not take effect at
  // runtime until the user re-saved the page.
  const serverKey =
    typeof agentSettings.acp_server === "string"
      ? agentSettings.acp_server
      : undefined;
  const effectiveModel = resolveEffectiveAcpModel({
    configured: agentSettings.acp_model as string | null | undefined,
    providerDefault: getAcpPreferredDefaultModel(serverKey),
  });
  if (effectiveModel) {
    payload.acp_model = effectiveModel;
  }

  return payload;
}

function buildConfiguredOpenHandsAgentSettings(
  settings: Settings,
  runtimeServicesInfo?: RuntimeServicesInfo | null,
): AgentSettingsPayload {
  const agentSettings = toRecord(settings.agent_settings);
  const llm = buildNormalizedLlmSettings(agentSettings.llm);

  // Stream assistant tokens (parity with ACP agents). The agent-server only
  // emits StreamingDeltaEvents for SDK LLM agents when an LLM has stream=True.
  llm.stream = true;

  const mcpConfig = toRecord(agentSettings.mcp_config);
  if (Object.keys(mcpConfig).length === 0) {
    delete agentSettings.mcp_config;
  }

  delete agentSettings.acp_server;
  for (const key of ACP_SETTINGS_KEYS) {
    delete agentSettings[key];
  }
  // ``acp_env`` is no longer a forwarded ACP setting (provider creds ride the
  // Secrets panel), but a legacy value may linger on persisted settings —
  // scrub it so it never leaks onto the OpenHands payload.
  delete agentSettings.acp_env;

  return {
    ...agentSettings,
    llm,
    agent_context: buildAgentContext(
      agentSettings,
      runtimeServicesInfo,
      settings.disabled_skills,
    ),
    tools: getAgentTools(agentSettings),
  };
}

function buildConfiguredAgentSettings(
  settings: Settings,
  runtimeServicesInfo?: RuntimeServicesInfo | null,
): AgentSettingsPayload {
  return isAcpAgent(settings)
    ? buildConfiguredAcpAgentSettings(settings, runtimeServicesInfo)
    : buildConfiguredOpenHandsAgentSettings(settings, runtimeServicesInfo);
}

function buildConfiguredConversationSettings(options: {
  settings: Settings;
  query?: string;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  workingDir?: string;
}): ConversationSettingsPayload {
  const { settings, query, conversationInstructions, plugins, workingDir } =
    options;
  const conversationSettings = toRecord(settings.conversation_settings);
  const initialMessage = buildInitialMessage(query, conversationInstructions);

  CONVERSATION_SETTINGS_METADATA_KEYS.forEach(
    (key) => delete conversationSettings[key],
  );

  const payload: ConversationSettingsPayload = {
    ...conversationSettings,
    workspace: {
      kind: "LocalWorkspace",
      working_dir: workingDir ?? getAgentServerWorkingDir(),
    },
    ...(initialMessage ? { initial_message: initialMessage } : {}),
    ...(plugins?.length
      ? {
          plugins: plugins.map((plugin) => ({
            source: plugin.source,
            ...(plugin.ref ? { ref: plugin.ref } : {}),
            ...(plugin.repo_path ? { repo_path: plugin.repo_path } : {}),
          })),
        }
      : {}),
  };

  return payload;
}

interface LookupSecret {
  kind: "LookupSecret";
  url: string;
  headers?: Record<string, string>;
  description?: string;
}

/** A custom secret's public identity — name + optional description, no value. */
type CustomSecretInput = { name: string; description?: string };

type StartConversationPayloadBase = Record<string, unknown> & {
  workspace: LocalWorkspacePayload;
  confirmation_policy: SettingsRecord;
  security_analyzer?: SettingsRecord;
  initial_message?: InitialMessagePayload;
  max_iterations: number;
  stuck_detection: true;
  autotitle: boolean;
  title_llm_profile?: string;
  worktree: boolean;
  secrets_encrypted?: true;
  conversation_id?: string;
  secrets?: Record<string, LookupSecret>;
  tags?: Record<string, string>;
  client_tools: ClientToolSpec[];
  tool_module_qualnames?: Record<string, string>;
};

type AgentSettingsStartConversationPayload = StartConversationPayloadBase & {
  // Omitted when launching via ``agent_profile_id`` — the two are mutually
  // exclusive agent sources; the server resolves the profile server-side.
  agent_settings?: AgentSettingsPayload;
  agent_profile_id?: string;
  agent?: never;
};

// The local planner bypasses agent assembly entirely and sends a raw agent
// spec, so neither of the two managed agent sources applies.
type RawAgentStartConversationPayload = StartConversationPayloadBase & {
  agent: SettingsRecord;
  agent_settings?: never;
  agent_profile_id?: never;
  parent_conversation_id?: string;
};

export interface StartConversationOptions {
  settings: Settings;
  query?: string;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  conversationId?: string;
  workingDir?: string;
  worktree?: boolean;
  encryptedAgentSettings?: Record<string, SettingsValue>;
  encryptedConversationSettings?: Record<string, SettingsValue>;
  secretsEncrypted?: boolean;
  customSecrets?: CustomSecretInput[];
  // When set, the conversation launches from this AgentProfile (resolved
  // server-side) instead of an inline ``agent_settings`` dump (#3727).
  agentProfileId?: string;
  agentProfileKind?: AgentKind;
  titleLlmProfile?: string;
  runtimeServicesInfo?: RuntimeServicesInfo | null;
}

/**
 * Build the `request.secrets` map shared by the standard and planning
 * conversation builders. Every saved secret rides as a LookupSecret the
 * agent-server resolves from its own store at spawn time — `request.secrets` is
 * the sole channel, uniform for ACP and non-ACP (agent-canvas#1039). For ACP the
 * resolution runs off the event loop (software-agent-sdk#3510, >=1.25.0), so the
 * loopback fetch can't deadlock. Returns `undefined` when there are no custom
 * secrets so callers can omit the field.
 */
function buildCustomSecrets(
  customSecrets: CustomSecretInput[] | undefined,
): Record<string, LookupSecret> | undefined {
  if (!customSecrets?.length) return undefined;

  const backend = getEffectiveLocalBackend();
  const headers = backend ? buildAuthHeaders(backend) : {};

  const secrets: Record<string, LookupSecret> = {};
  for (const secret of customSecrets) {
    const lookupSecret: LookupSecret = {
      kind: "LookupSecret",
      url: `/api/settings/secrets/${encodeURIComponent(secret.name)}`,
      description: secret.description,
    };
    if (Object.keys(headers).length > 0) {
      lookupSecret.headers = headers;
    }
    secrets[secret.name] = lookupSecret;
  }
  return secrets;
}

export function buildStartConversationRequest(
  options: StartConversationOptions,
): AgentSettingsStartConversationPayload {
  const sourceAgentSettings = options.encryptedAgentSettings
    ? { ...options.settings, agent_settings: options.encryptedAgentSettings }
    : options.settings;

  const acpMode = isAcpAgent(sourceAgentSettings);
  const launchAgentKind = options.agentProfileId
    ? options.agentProfileKind
    : acpMode
      ? "acp"
      : "openhands";
  const agentSettings = buildConfiguredAgentSettings(
    sourceAgentSettings,
    options.runtimeServicesInfo,
  );
  const acpServerTag = acpMode
    ? getAcpServerTag(sourceAgentSettings)
    : undefined;

  const sourceConversationOptions = options.encryptedConversationSettings
    ? {
        ...options,
        settings: {
          ...options.settings,
          conversation_settings: options.encryptedConversationSettings,
        },
      }
    : options;

  const conversationSettings = buildConfiguredConversationSettings(
    sourceConversationOptions,
  );

  const payload: AgentSettingsStartConversationPayload = {
    // ``agent_profile_id`` and ``agent_settings`` are mutually exclusive agent
    // sources; the profile path lets the server resolve the profile (#3727).
    //
    // Enrichment boundary: on the profile path the server rebuilds the agent
    // purely from the stored profile fields, so the client-owned enrichments
    // this adapter folds into ``agent_settings`` do NOT apply. The exec toolset
    // (terminal/file_editor/task_tracker) and public-skill loading are the
    // server/SDK's responsibility to restore on the profile path — tracked in
    // software-agent-sdk#3967 (profile resolution must attach the default
    // toolset + public skills, else a profile-launched OpenHands agent has only
    // Finish/Think). The dev ``RUNTIME_SERVICES`` system-message suffix remains
    // agent-settings-only; the Canvas UI tool is a top-level client tool and
    // therefore works on both inline-agent and profile launch paths.
    //
    // Persistent memory is NOT on that boundary: ``load_memory`` is a global
    // user preference, so the agent-server stamps the stored
    // ``agent_settings.agent_context.load_memory`` onto the profile-resolved
    // agent the same way it already applies the global ``mcp_config``. The
    // toggle therefore applies to both launch paths, and the client must not
    // re-send it here (``agent_profile_id`` and ``agent_settings`` are
    // mutually exclusive).
    ...(options.agentProfileId
      ? { agent_profile_id: options.agentProfileId }
      : { agent_settings: agentSettings }),
    workspace: conversationSettings.workspace,
    client_tools:
      launchAgentKind === "openhands" ? [CANVAS_UI_CLIENT_TOOL] : [],
    confirmation_policy:
      getConversationConfirmationPolicy(conversationSettings),
    max_iterations: resolveMaxIterations(conversationSettings.max_iterations),
    stuck_detection: true,
    autotitle: true,
    ...(options.titleLlmProfile
      ? { title_llm_profile: options.titleLlmProfile }
      : {}),
    worktree: options.worktree ?? true,
  };

  // A profile launch resolves the ACP server server-side, so don't stamp the
  // tag from current settings (it may not match the launched profile).
  if (!options.agentProfileId && acpServerTag) {
    payload.tags = { [ACP_SERVER_TAG_KEY]: acpServerTag };
  }

  // ``secrets_encrypted`` makes the agent-server decrypt request secrets at
  // conversation start. Non-ACP conversations need it for encrypted LLM keys.
  // ACP normally carries provider credentials as LookupSecrets, so avoid
  // forcing a cipher on fresh ACP-only backends. The exception is MCP:
  // encrypted settings round-trip mcp_config secrets as Fernet tokens,
  // and ACP forwards mcp_config directly to the subprocess.
  if (
    !options.agentProfileId &&
    options.secretsEncrypted &&
    (!acpMode || hasEncryptedMcpSecrets(agentSettings.mcp_config))
  ) {
    payload.secrets_encrypted = true;
  }

  if (options.conversationId) {
    payload.conversation_id = options.conversationId;
  }

  const securityAnalyzer =
    getConversationSecurityAnalyzer(conversationSettings);
  if (securityAnalyzer) {
    payload.security_analyzer = securityAnalyzer;
  }

  if (conversationSettings.initial_message) {
    payload.initial_message = conversationSettings.initial_message;
  }

  if (conversationSettings.plugins) {
    payload.plugins = conversationSettings.plugins;
  }

  if (conversationSettings.hook_config) {
    payload.hook_config = conversationSettings.hook_config;
  }

  const toolModuleQualnames = {
    ...((conversationSettings.tool_module_qualnames as
      | Record<string, string>
      | undefined) ?? {}),
  };
  delete toolModuleQualnames[LEGACY_CANVAS_UI_TOOL_NAME];
  delete toolModuleQualnames[CANVAS_UI_CLIENT_TOOL_NAME];
  if (Object.keys(toolModuleQualnames).length > 0) {
    payload.tool_module_qualnames = toolModuleQualnames;
  }

  if (conversationSettings.agent_definitions) {
    payload.agent_definitions = conversationSettings.agent_definitions;
  }

  const secrets = buildCustomSecrets(options.customSecrets);
  if (secrets) {
    payload.secrets = secrets;
  }

  return payload;
}

export function buildStartPlanningConversationRequest(options: {
  encryptedAgentSettings: Record<string, SettingsValue>;
  workingDir: string;
  parentConversationId: string;
  initialMessage?: string;
  secretsEncrypted?: boolean;
  customSecrets?: CustomSecretInput[];
  /** Mirrors the parent's configured `conversation_settings.max_iterations` — see DEFAULT_MAX_ITERATIONS. */
  maxIterations?: number;
  /** Mirrors the code agent's skill filtering — see buildConfiguredOpenHandsAgentSettings. */
  disabledSkills?: string[];
}): RawAgentStartConversationPayload {
  const agentSettings = toRecord(options.encryptedAgentSettings);
  const llm = buildNormalizedLlmSettings(agentSettings.llm);
  // Stream assistant tokens, matching the code agent (see
  // buildConfiguredOpenHandsAgentSettings) — otherwise the agent-server never
  // emits StreamingDeltaEvents for the planner and its replies would appear
  // all at once instead of token-by-token.
  llm.stream = true;

  const planPath = buildPlanPath(options.workingDir);

  // Put the planner's directive + boundaries in the system prompt (matching the
  // OpenHands app-server's PLANNING_AGENT_INSTRUCTION), preserving any suffix
  // buildAgentContext already set (e.g. the runtime-services block).
  const agentContext = buildAgentContext(
    agentSettings,
    undefined,
    options.disabledSkills,
  );
  const existingSuffix = agentContext.system_message_suffix;
  agentContext.system_message_suffix =
    typeof existingSuffix === "string"
      ? `${PLANNING_AGENT_INSTRUCTION}\n\n${existingSuffix}`
      : PLANNING_AGENT_INSTRUCTION;

  // Idle planner: "Create a Plan" only switches to plan mode and provisions
  // this conversation; the user sends the first message themselves, so nothing
  // is injected into the chat. An explicit initialMessage is still honored.
  const initialMessage = buildInitialMessage(options.initialMessage);

  const payload: RawAgentStartConversationPayload = {
    agent: {
      kind: "Agent",
      llm,
      tools: [
        { name: "glob", params: {} },
        { name: "grep", params: {} },
        {
          name: PLANNING_FILE_EDITOR_TOOL_NAME,
          params: { plan_path: planPath },
        },
      ],
      system_prompt_filename: PLANNING_SYSTEM_PROMPT_FILENAME,
      system_prompt_kwargs: { plan_structure: PLAN_STRUCTURE_TEXT },
      // agent_context (skills + project context) is intentionally included so
      // the planner shares the code agent's project context. This goes beyond
      // the bare SDK `get_planning_agent()` preset (which sets no context); it
      // mirrors how the frontend builds every agent, so the local planner is
      // not context-starved relative to the code agent.
      agent_context: agentContext,
      // Mirror the SDK planning preset's condenser (openhands-tools preset
      // `get_planning_condenser`): a larger rolling window and more pinned
      // initial context than the default, with its own usage_id so its
      // summarization LLM calls are attributed separately. The planning
      // conversation is sent as a raw agent spec (it bypasses the agent-server's
      // default agent assembly), so without this it would never condense —
      // diverging from the canonical planning agent on long sessions.
      condenser: {
        kind: "LLMSummarizingCondenser",
        llm: { ...llm, usage_id: "planning_condenser" },
        max_size: 100,
        keep_first: 6,
      },
    },
    workspace: {
      kind: "LocalWorkspace",
      working_dir: options.workingDir,
    },
    // No Canvas UI client tool: the planner's only output is PLAN.md, which the
    // Planner tab surfaces on its own via PlanningFileEditorObservation. Handing
    // it the panel-control tool would only let it navigate the right-side panel
    // away from the plan the user is reading.
    client_tools: [],
    confirmation_policy: { kind: "NeverConfirm" },
    max_iterations: options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    stuck_detection: true,
    autotitle: false,
    worktree: false,
    // Two independent links to the parent, for two different jobs:
    //
    // - ``parent_conversation_id`` is the server-owned relationship (SDK
    //   #4188, agent-server >= 1.37.1). The parent's ``sub_conversation_ids``
    //   is derived from it, so the planner is recoverable from the server on
    //   any browser and after storage loss. Older agent-servers ignore the
    //   field (``StartConversationRequest`` does not forbid extras), which is
    //   why the client-side metadata hint is still written as a fallback.
    // - The tag is what hides the helper from the conversation list; it is
    //   also the only marker on agent-servers too old for the parent link.
    parent_conversation_id: options.parentConversationId,
    tags: { [LOCAL_PLANNER_PARENT_TAG_KEY]: options.parentConversationId },
    ...(initialMessage ? { initial_message: initialMessage } : {}),
  };

  if (options.secretsEncrypted) {
    payload.secrets_encrypted = true;
  }

  const secrets = buildCustomSecrets(options.customSecrets);
  if (secrets) {
    payload.secrets = secrets;
  }

  return payload;
}

/**
 * Resolve the encrypted LLM config for a named LLM profile, so a planner can
 * run the parent conversation's *current* model (`active_profile`, tracking
 * `/model` and `SwitchLLMTool`) rather than a stale launch-time value.
 * Returns `null` — caller falls back further — if unknown or lookup fails.
 */
async function resolveLlmProfileSettings(
  profileName: string,
): Promise<SettingsRecord | null> {
  try {
    const { default: ProfilesService } =
      await import("./profiles-service/profiles-service.api");
    // ``encrypted`` matches ``secrets_encrypted`` on the payload: the
    // agent-server decrypts with the same cipher it encrypted with.
    const detail = await ProfilesService.getProfile(profileName, "encrypted");
    return isPlainRecord(detail.config) ? detail.config : null;
  } catch (error) {
    console.warn(
      `Falling back: could not resolve LLM profile ${profileName}`,
      error,
    );
    return null;
  }
}

/**
 * Resolve the encrypted LLM config the given AgentProfile launches with, as a
 * fallback for `resolveLlmProfileSettings`. AgentProfiles hold a reference
 * (`llm_profile_ref`), not credentials, so this is a two-hop lookup: profile
 * id -> llm profile name -> encrypted LLM config. Returns `null` — caller
 * falls back to global settings — for an unknown/ACP/dangling-ref profile or
 * a failed lookup, none of which should block plan creation outright.
 */
async function resolveAgentProfileLlmSettings(
  agentProfileId: string,
): Promise<SettingsRecord | null> {
  try {
    const { default: AgentProfilesService } =
      await import("./agent-profiles-service/agent-profiles-service.api");

    const { profiles } = await AgentProfilesService.listProfiles();
    const summary = profiles.find(
      (profile) => profile.id === agentProfileId && profile.llm_profile_ref,
    );
    if (!summary?.llm_profile_ref) return null;

    return await resolveLlmProfileSettings(summary.llm_profile_ref);
  } catch (error) {
    console.warn(
      `Falling back to global agent settings: could not resolve the LLM for agent profile ${agentProfileId}`,
      error,
    );
    return null;
  }
}

export async function buildStartPlanningConversationRequestWithEncryptedSettings(options: {
  workingDir: string;
  parentConversationId: string;
  /**
   * The parent conversation's current LLM profile (`AppConversation.active_profile`,
   * tracking `/model` and `SwitchLLMTool`). Takes priority over
   * `parentAgentProfileId` so a model switch on the parent carries over to a
   * planner created afterward, without a *different* conversation's global
   * profile activation repointing it.
   */
  parentActiveProfileName?: string | null;
  /**
   * `launched_agent_profile.agent_profile_id` of the parent, when started
   * from an AgentProfile. Fallback for when `parentActiveProfileName` can't
   * be resolved (e.g. an ACP parent, whose `active_profile` is a stale
   * launch-time snapshot rather than anything the ACP agent itself runs).
   */
  parentAgentProfileId?: string | null;
  initialMessage?: string;
}): Promise<RawAgentStartConversationPayload> {
  const { SecretsService } = await import("./secrets-service");

  const [settingsResult, customSecrets] = await Promise.all([
    SettingsService.getSettingsForConversation(),
    SecretsService.getSecrets(),
  ]);

  const profileLlm =
    (options.parentActiveProfileName
      ? await resolveLlmProfileSettings(options.parentActiveProfileName)
      : null) ??
    (options.parentAgentProfileId
      ? await resolveAgentProfileLlmSettings(options.parentAgentProfileId)
      : null);
  const encryptedAgentSettings: Record<string, SettingsValue> = profileLlm
    ? { ...settingsResult.agentSettings, llm: profileLlm as SettingsValue }
    : settingsResult.agentSettings;

  await assertSubscriptionAuthReady(encryptedAgentSettings);

  // Mirror the code agent's configured cap (see DEFAULT_MAX_ITERATIONS) rather
  // than hardcoding a different value the planner could silently stop against.
  const maxIterations = resolveMaxIterations(
    settingsResult.conversationSettings.max_iterations,
  );

  return buildStartPlanningConversationRequest({
    ...options,
    encryptedAgentSettings,
    secretsEncrypted: settingsResult.secretsEncrypted,
    customSecrets,
    maxIterations,
    disabledSkills: settingsResult.disabledSkills,
  });
}

export const SUBSCRIPTION_LOGIN_REQUIRED_ERROR =
  "Connect your ChatGPT subscription before starting a conversation with this LLM profile.";

/**
 * Throws if a ChatGPT subscription LLM profile is not connected.
 * Called before conversation creation and LLM profile switch only — not on
 * subsequent message sends or conversation resume. The agent-server must handle
 * mid-conversation token expiry gracefully.
 */
export async function assertSubscriptionAuthReady(
  agentSettings: Record<string, unknown>,
): Promise<void> {
  const llm = toRecord(agentSettings.llm);
  if (!isSubscriptionLlmConfig(llm)) return;

  const status = await LLMSubscriptionService.getOpenAIStatus();
  if (!status.connected) {
    throw new Error(SUBSCRIPTION_LOGIN_REQUIRED_ERROR);
  }
}

export async function buildStartConversationRequestWithEncryptedSettings(options: {
  settings: Settings;
  query?: string;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  conversationId?: string;
  workingDir?: string;
  worktree?: boolean;
  agentProfileId?: string;
  agentProfileKind?: AgentKind;
  titleLlmProfile?: string;
}): Promise<Record<string, unknown>> {
  const { SecretsService } = await import("./secrets-service");

  const [settingsResult, customSecrets, runtimeServicesInfo] =
    await Promise.all([
      SettingsService.getSettingsForConversation(),
      SecretsService.getSecrets(),
      fetchBackendRuntimeServicesInfo(),
    ]);

  const { agentSettings, conversationSettings, secretsEncrypted } =
    settingsResult;

  // A profile launch resolves the LLM server-side, so the current-settings
  // subscription check doesn't apply (and can't see the profile's LLM).
  if (!options.agentProfileId) {
    await assertSubscriptionAuthReady(agentSettings);
  }

  return buildStartConversationRequest({
    ...options,
    encryptedAgentSettings: agentSettings,
    encryptedConversationSettings: conversationSettings,
    secretsEncrypted,
    customSecrets,
    runtimeServicesInfo,
  });
}

export function emptyHooksResponse(): GetHooksResponse {
  return { hooks: [] };
}
