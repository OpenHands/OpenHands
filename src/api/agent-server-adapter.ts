import i18n, { OPENHANDS_I18N_NAMESPACE } from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { hasPluginParameters } from "#/utils/plugin-parameters";
import { HOOK_EVENT_FIELDS, HookType } from "@openhands/typescript-client";
import type {
  ACPModelInfo,
  ConversationRuntimeInfo,
  HookConfig,
} from "@openhands/typescript-client";
import { ServerClient } from "@openhands/typescript-client/clients";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  AgentKind,
  Settings,
  SettingsValue,
  isProvider,
} from "#/types/settings";
import {
  resolveEffectiveAcpModel,
  toAcpModelOptions,
} from "#/constants/acp-providers";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import {
  getCachedAgentServerInfo,
  type AgentServerInfo,
} from "./agent-server-compatibility";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { buildAuthHeaders } from "./backend-registry/auth";
import {
  GetHooksResponse,
  PluginSpec,
  AppConversation,
  AppConversationPage,
  RuntimeConversationStats,
  SandboxStatus,
} from "./conversation-service/agent-server-conversation-service.types";
import { combineUsageMetrics } from "#/utils/conversation-metrics";
import {
  AUTOMATION_DRAFT_ID_TAG_KEY,
  AUTOMATION_EDIT_ID_TAG_KEY,
  AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY,
  AUTOMATION_SETUP_TAG_KEY,
  buildAutomationSetupModeTags,
} from "#/utils/automation-draft-tags";
import {
  buildSkillEnablementFilter,
  findInvokedCatalogSkill,
  profileSkillEnablement,
  type SkillEnablement,
} from "#/utils/skill-enablement";
import type { StoredOpenHandsProfile } from "./agent-profiles-service/profile-behavior";
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
  LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
} from "./launch-child-conversation-client-tool";
import { AUTOMATION_FORM_UPDATE_CLIENT_TOOL } from "./automation-form-client-tool";
import { REQUEST_SECRET_CLIENT_TOOL } from "./request-secret-client-tool";
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
  /** Availability of the runtime that backs a local conversation. */
  runtime_info?: Pick<
    ConversationRuntimeInfo,
    "runtime_status" | "can_resume"
  > | null;
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
  /**
   * Raw per-usage-id LLM stats from the agent-server. Search/list responses
   * often carry real usage here even when `metrics` above comes back unset;
   * {@link toAppConversation} combines this as a fallback in that case.
   */
  stats?: RuntimeConversationStats | null;
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
  available_models?: ACPModelInfo[] | null;
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
    secret_refs?: string[] | null;
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

// Falls back to the same default the code agent uses when the user has not
// configured `conversation_settings.max_iterations` (see buildConfiguredConversationSettings).
const DEFAULT_MAX_ITERATIONS = 500;

function resolveMaxIterations(value: unknown): number {
  return typeof value === "number" ? value : DEFAULT_MAX_ITERATIONS;
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

async function fetchBackendServerInfo(): Promise<AgentServerInfo | null> {
  let clientOptions: ReturnType<typeof getAgentServerClientOptions>;
  try {
    clientOptions = getAgentServerClientOptions({ timeout: 3000 });
  } catch {
    return null;
  }

  const cached = getCachedAgentServerInfo({ host: clientOptions.host });
  if (cached) return cached;

  try {
    return (await new ServerClient(
      clientOptions,
    ).getServerInfo()) as AgentServerInfo;
  } catch {
    return null;
  }
}

export async function fetchBackendExecutionRuntime(): Promise<
  AgentServerInfo["execution_runtime"]
> {
  const serverInfo = await fetchBackendServerInfo();
  return serverInfo?.execution_runtime;
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

/**
 * System-message instruction appended when the user enables "Run on first
 * message" for the Model Router. With an active meta-profile the
 * agent-server attaches the ``route_task_to_model`` tool; this suffix tells
 * the agent to call it with the *first* user message before doing anything
 * else, so the opening request is always routed to the classifier-selected
 * model. Subsequent messages are left to the agent's own judgment.
 *
 * Only emitted when ``runAtStart`` is on AND a meta-profile is active
 * (``hasActiveRouter``): with no active meta-profile the agent-server does
 * not attach ``route_task_to_model``, so emitting the instruction would tell
 * the agent to call a tool it does not have. Gating here — inside the single
 * suffix builder that every launch path funnels through — keeps behavior
 * consistent regardless of how the toggle was persisted, so a stale ``true``
 * (e.g. the active profile was just deleted) can never reach the prompt.
 *
 * Returns `undefined` when disabled so no prompt noise is added — the default
 * "everything works as normal" path is byte-identical to before.
 */
export function buildRouterAtStartSystemSuffix(
  runAtStart: boolean,
  hasActiveRouter: boolean,
): string | undefined {
  if (!runAtStart || !hasActiveRouter) return undefined;
  return [
    "<ROUTE_AT_CONVERSATION_START>",
    "Before responding to the FIRST user message of this conversation, call",
    "the `route_task_to_model` tool with that user message so the active",
    "Model Router selects the appropriate model. After the router switches",
    "the model, continue handling the user's request with the switched",
    "model. Apply this only to the very first user message; do not route",
    "subsequent messages this way.",
    "</ROUTE_AT_CONVERSATION_START>",
  ].join("\n");
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
    selected_repository:
      metadata?.selected_repository ?? info.tags?.[REPOSITORY_TAG_KEY] ?? null,
    selected_branch:
      metadata?.selected_branch ?? info.tags?.[SELECTED_BRANCH_TAG_KEY] ?? null,
    git_provider:
      metadata?.git_provider ??
      (isProvider(info.tags?.[GIT_PROVIDER_TAG_KEY])
        ? info.tags[GIT_PROVIDER_TAG_KEY]
        : null),
    // Prefer the creating client's localStorage metadata; fall back to the
    // server-side workspace tag so local sidebar grouping survives on another
    // browser, device, or after cleared site data (#17386).
    selected_workspace:
      metadata?.selected_workspace ?? info.tags?.[WORKSPACE_TAG_KEY] ?? null,
    active_profile: metadata?.active_profile ?? null,
    title: info.title?.trim()
      ? info.title
      : getDefaultConversationTitle(info.id),
    trigger: null,
    pr_number: [],
    agent_kind: isAcp ? "acp" : "openhands",
    acp_server: acpServer,
    acp_available_models: isAcp
      ? toAcpModelOptions(info.available_models)
      : null,
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
      : combineUsageMetrics(info.stats),
    created_at: info.created_at,
    updated_at: info.updated_at,
    execution_status:
      (info.execution_status as AppConversation["execution_status"]) ??
      ExecutionStatus.IDLE,
    sandbox_status:
      info.runtime_info?.runtime_status === "missing" &&
      !info.runtime_info.can_resume
        ? "MISSING"
        : ((info.sandbox_status as SandboxStatus | null) ?? null),
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

interface LocalWorkspacePayload {
  kind: "LocalWorkspace";
  working_dir: string;
}

const DOCKER_EXECUTION_WORKING_DIR = "/workspace" as const;

interface DockerExecutionWorkspacePayload {
  kind: "DockerExecutionWorkspace";
  working_dir: typeof DOCKER_EXECUTION_WORKING_DIR;
}

type WorkspacePayload = LocalWorkspacePayload | DockerExecutionWorkspacePayload;

interface InitialMessagePayload {
  role: "user";
  content: Array<{ type: "text"; text: string }>;
  run: true;
}

type ConversationSettingsPayload = SettingsRecord & {
  workspace: WorkspacePayload;
  initial_message?: InitialMessagePayload;
};

export const ACP_SERVER_TAG_KEY = "acpserver";
export const CLIENT_SOURCE_TAG_KEY = "clientsource";
export const AGENT_CANVAS_SOURCE = "agentcanvas";

/**
 * Server-side conversation tag that persists the home-page workspace selection
 * on the agent-server, so local sidebar grouping survives a different browser,
 * device, or cleared localStorage (the localStorage metadata store stays the
 * warm cache for the creating client). See #17386.
 *
 * The agent-server validates tag keys against `^[a-z0-9]+$` (SDK
 * `conversation/types.py`), so the key must stay lowercase alphanumeric with
 * no separators — an underscored key like `selected_workspace` would make the
 * create request 422. It is already in {@link RESERVED_CONVERSATION_TAG_KEYS},
 * so it never appears as an editable chip.
 */
export const WORKSPACE_TAG_KEY = "workspace";

/**
 * Build the workspace tag to stamp at conversation creation, normalizing the
 * path and returning an empty object when there is nothing to persist (so the
 * caller can skip adding a `tags` payload entirely).
 */
export function buildWorkspaceTags(
  selectedWorkspace?: string | null,
): Record<string, string> {
  const workspace = selectedWorkspace?.trim().replace(/\/+$/, "");
  return workspace ? { [WORKSPACE_TAG_KEY]: workspace } : {};
}

// Repository tags, read back like the workspace tag (#15598). Keys must match
// the agent-server's `^[a-z0-9]+$`.
export const REPOSITORY_TAG_KEY = "repository";
export const SELECTED_BRANCH_TAG_KEY = "selectedbranch";
export const GIT_PROVIDER_TAG_KEY = "gitprovider";

export const AUTOMATION_TRIGGER_TAG_KEY = "automationtrigger";
export const AUTOMATION_ID_TAG_KEY = "automationid";
export const AUTOMATION_NAME_TAG_KEY = "automationname";
export const AUTOMATION_RUN_ID_TAG_KEY = "automationrunid";

/**
 * Tag keys stamped on conversations created by automation runs (see the SDK's
 * `RemoteWorkspace.default_conversation_tags`). The presence of any of these
 * marks a conversation as automation-born.
 */
export const AUTOMATION_TAG_KEYS: readonly string[] = [
  AUTOMATION_TRIGGER_TAG_KEY,
  AUTOMATION_ID_TAG_KEY,
  AUTOMATION_NAME_TAG_KEY,
  AUTOMATION_RUN_ID_TAG_KEY,
];

/**
 * Conversation tag keys that must not appear as generic chips / hovercard
 * rows. Each is either already surfaced by a first-class UI source or is
 * internal routing data:
 * - ``acpserver`` → ACP provider chip
 * - ``clientsource`` → telemetry attribution
 * - ``title`` → conversation card heading
 * - git / repo / branch / workspace stamps → repo-branch metadata + directory
 *   footer / hovercard rows (``selected_repository``, ``selected_branch``,
 *   ``git_provider``, ``workspace.working_dir``)
 * - the automation family (``automationtrigger`` / ``automationid`` /
 *   ``automationname`` / ``automationrunid``) → provenance the SDK stamps at
 *   creation; the conversation panel's automation filter is its first-class
 *   UI source. The tag surface is user organization data, so machine stamps
 *   stay out of it — and users can't edit or spoof automation classification.
 * - ``localplannerparent`` → internal routing for the local planner; already
 *   surfaced by the hidden-from-list planner filter
 * - ``automationsetup`` → user-visible setup-mode marker so users can find
 *   setup conversations and filter by them
 * - ``automationdraftid`` / ``automationmaterializeddraftid`` → internal
 *   routing for resuming server-backed automation setup drafts (the last one
 *   links a resumed conversation to the automation materialized from its draft)
 * - ``automationeditid`` → the saved automation this setup page is editing
 */
export const RESERVED_CONVERSATION_TAG_KEYS: ReadonlySet<string> = new Set([
  ACP_SERVER_TAG_KEY,
  CLIENT_SOURCE_TAG_KEY,
  AUTOMATION_TRIGGER_TAG_KEY,
  AUTOMATION_ID_TAG_KEY,
  AUTOMATION_NAME_TAG_KEY,
  AUTOMATION_RUN_ID_TAG_KEY,
  AUTOMATION_DRAFT_ID_TAG_KEY,
  AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY,
  AUTOMATION_EDIT_ID_TAG_KEY,
  "title",
  GIT_PROVIDER_TAG_KEY,
  "git_provider",
  "repo_name",
  "repo",
  "repository",
  SELECTED_BRANCH_TAG_KEY,
  "selected_branch",
  "branch",
  "archiveworkspacepath",
  "workspace",
  "working_dir",
  // `pinned` / `archived` are the sidebar's pin/archive flags (see
  // `conversation-tag-flags.ts`), surfaced by the pin icon and the archived
  // presentation rather than as generic tag chips.
  "pinned",
  "archived",
  LOCAL_PLANNER_PARENT_TAG_KEY,
  // The launched agent profile, which the app server stamps as tags; the
  // hovercard names the profile instead of echoing the raw id and revision.
  "agentprofileid",
  "agentprofilerevision",
]);

/**
 * High-signal tag keys shown first in the chip row (before A–Z). Automation
 * setup mode should be easy to spot and filter; automations often stamp
 * ``origin``; remaining free-form tags sort alphabetically.
 */
export const PRIORITY_CONVERSATION_TAG_KEYS: readonly string[] = [
  AUTOMATION_SETUP_TAG_KEY,
  "origin",
];

/**
 * User-facing subset of a conversation's server-side tags: everything except
 * {@link RESERVED_CONVERSATION_TAG_KEYS}, as stable ``[key, value]`` entries.
 * Priority keys come first (in {@link PRIORITY_CONVERSATION_TAG_KEYS} order);
 * the rest sort A–Z so chip order doesn't shuffle between refetches.
 */
export function getDisplayConversationTags(
  tags: Record<string, string> | null | undefined,
): Array<[string, string]> {
  if (!tags) {
    return [];
  }
  // Both the reserved-key check and the priority lookup must see the same
  // normalized key: a cloud backend can stamp ``Origin`` / `` origin``, and
  // ranking those off the raw key would silently drop them out of first place.
  const priorityRank = (key: string): number => {
    const index = PRIORITY_CONVERSATION_TAG_KEYS.indexOf(
      key.trim().toLowerCase(),
    );
    return index === -1 ? Number.POSITIVE_INFINITY : index;
  };

  return Object.entries(tags)
    .filter(
      ([key, value]) =>
        !RESERVED_CONVERSATION_TAG_KEYS.has(key.trim().toLowerCase()) &&
        typeof value === "string" &&
        // Bare tags (empty value) are displayable — chips/tooltips render
        // the key. Whitespace-only values stay dropped (raw-write junk).
        (value === "" || value.trim().length > 0),
    )
    .sort(([a], [b]) => {
      const aRank = priorityRank(a);
      const bRank = priorityRank(b);
      if (aRank !== bRank) {
        return aRank - bRank;
      }
      return a.localeCompare(b);
    });
}

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
  } else {
    delete llm.auth_type;
    delete llm.subscription_vendor;
  }

  return llm;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
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

/**
 * The Settings → Runtime options a start request carries. Only non-defaults
 * are sent, so servers without the fields see the request they always did.
 */
function getConversationRuntimeOptions(conversationSettings: SettingsRecord) {
  const {
    worktree_base: base,
    worktree_base_ref: ref,
    worktree_branch_prefix: branchPrefix,
    container_memory: memory,
    container_cpus: cpus,
    container_browser: browser,
  } = conversationSettings;
  const worktreeOptions = {
    ...(typeof base === "string" && base !== "default_branch" ? { base } : {}),
    ...(base === "ref" && typeof ref === "string" ? { ref } : {}),
    ...(typeof branchPrefix === "string" && branchPrefix !== "openhands/"
      ? { branch_prefix: branchPrefix }
      : {}),
  };
  const container = {
    ...(typeof memory === "string" && memory ? { memory } : {}),
    ...(typeof cpus === "number" ? { cpus } : {}),
    ...(browser === false ? { browser } : {}),
  };
  return {
    ...(Object.keys(worktreeOptions).length > 0
      ? { worktree_options: worktreeOptions }
      : {}),
    ...(Object.keys(container).length > 0 ? { container } : {}),
  };
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
  enablement: SkillEnablement = {},
): SettingsRecord {
  const existingContext = toRecord(agentSettings.agent_context);

  // Merge bundled public skills with any skills already present in the
  // agent context (e.g. user-defined skills set via the settings API).
  const existingSkills = Array.isArray(existingContext.skills)
    ? (existingContext.skills as SettingsRecord[])
    : [];
  const disabledSkills = enablement.disabledSkills ?? [];
  const disabledSkillNames = new Set(disabledSkills);
  const isSkillEnabled = buildSkillEnablementFilter(enablement);

  // The bundled catalog is allow-listed, not deny-listed: it is a build-time
  // snapshot of ~60 skills, so a deny-list puts every future addition into
  // every system prompt (OpenHands#16302). Skills the agent context already
  // carries are user-authored and stay opt-out.
  const mergedSkills = [
    ...existingSkills.filter(
      (skill) =>
        typeof skill.name !== "string" || !disabledSkillNames.has(skill.name),
    ),
    ...buildBundledSkills().filter((skill) => isSkillEnabled(skill.name)),
  ];

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
    // The backend also auto-loads user/project skills; the deny-list must
    // travel with the context so those skills are excluded from the system
    // prompt too. The allow-list has no counterpart to send: the backend
    // loads no catalog skills of its own (`load_public_skills` is false).
    disabled_skills: disabledSkills,
  };
}

function buildConfiguredConversationSettings(options: {
  settings: Settings;
  query?: string;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  workingDir?: string;
  executionRuntime?: AgentServerInfo["execution_runtime"];
}): ConversationSettingsPayload {
  const {
    settings,
    query,
    conversationInstructions,
    plugins,
    workingDir,
    executionRuntime,
  } = options;
  if (hasPluginParameters(plugins)) {
    throw new Error(
      i18n.t(I18nKey.LAUNCH$PARAMETERS_UNSUPPORTED_LOCAL, {
        ns: OPENHANDS_I18N_NAMESPACE,
      }),
    );
  }
  const conversationSettings = toRecord(settings.conversation_settings);
  const initialMessage = buildInitialMessage(query, conversationInstructions);

  CONVERSATION_SETTINGS_METADATA_KEYS.forEach(
    (key) => delete conversationSettings[key],
  );

  const workspace: WorkspacePayload =
    executionRuntime === "docker"
      ? {
          kind: "DockerExecutionWorkspace",
          working_dir: DOCKER_EXECUTION_WORKING_DIR,
        }
      : {
          kind: "LocalWorkspace",
          working_dir: workingDir ?? getAgentServerWorkingDir(),
        };
  const payload: ConversationSettingsPayload = {
    ...conversationSettings,
    workspace,
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
export type CustomSecretInput = { name: string; description?: string };

type StartConversationPayloadBase = Record<string, unknown> & {
  workspace: WorkspacePayload;
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
  parent_conversation_id?: string;
  secrets?: Record<string, LookupSecret>;
  tags?: Record<string, string>;
  client_tools: ClientToolSpec[];
  tool_module_qualnames?: Record<string, string>;
};

type ProfileStartConversationPayload = StartConversationPayloadBase & {
  agent_profile_id: string;
  agent_launch_additions?: {
    // Appended to the profile-resolved agent's system-message suffix.
    system_message_suffix_append?: string;
    // Runs the profile with this LLM profile instead of its own.
    llm_profile_ref?: string;
    // Merged into the profile-resolved agent's skills by name.
    skills?: BundledSkill[];
  };
  agent?: never;
  agent_settings?: never;
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
  automationSetup?: boolean;
  automationSetupTags?: Record<string, string>;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  conversationId?: string;
  // Links the new conversation to an existing one as its child. The
  // agent-server requires the parent to exist and to share this
  // conversation's requested `workspace.working_dir` (software-agent-sdk
  // #4188, agent-server >= 1.37.1); older servers ignore the field.
  parentConversationId?: string;
  workingDir?: string;
  worktree?: boolean;
  encryptedConversationSettings?: Record<string, SettingsValue>;
  customSecrets?: CustomSecretInput[];
  /** The AgentProfile the server builds the agent from (#3727). */
  agentProfileId: string;
  agentProfileKind?: AgentKind;
  /** LLM profile the agent profile runs with instead of its own. */
  llmProfileOverride?: string;
  /** The profile carries a Model Router, so router-at-start can apply. */
  profileHasRouter?: boolean;
  /** The ACP profile's server, stamped as the `acpserver` tag. */
  acpServer?: string;
  titleLlmProfile?: string;
  runtimeServicesInfo?: RuntimeServicesInfo | null;
  executionRuntime?: AgentServerInfo["execution_runtime"];
  workspaceHookConfig?: HookConfig | null;
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
export function buildCustomSecrets(
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
): ProfileStartConversationPayload {
  const launchSuffix = [
    buildRuntimeServicesSystemSuffix(options.runtimeServicesInfo),
    buildRouterAtStartSystemSuffix(
      options.settings.run_router_at_conversation_start ?? false,
      options.profileHasRouter ?? false,
    ),
  ]
    .filter((suffix): suffix is string => Boolean(suffix))
    .join("\n\n");
  // A catalog skill the opening message invokes loads for this conversation
  // even when the profile does not select it.
  const invokedSkillName =
    options.agentProfileKind === "openhands"
      ? findInvokedCatalogSkill(options.query)
      : undefined;
  const invokedSkill = invokedSkillName
    ? buildBundledSkills().find((skill) => skill.name === invokedSkillName)
    : undefined;
  const launchAdditions = {
    ...(launchSuffix ? { system_message_suffix_append: launchSuffix } : {}),
    ...(options.llmProfileOverride
      ? { llm_profile_ref: options.llmProfileOverride }
      : {}),
    ...(invokedSkill ? { skills: [invokedSkill] } : {}),
  };

  const sourceConversationOptions = options.encryptedConversationSettings
    ? {
        ...options,
        settings: {
          ...options.settings,
          conversation_settings: options.encryptedConversationSettings,
        },
      }
    : options;

  const conversationSettings = buildConfiguredConversationSettings({
    ...sourceConversationOptions,
    executionRuntime: options.executionRuntime,
  });

  const payload: ProfileStartConversationPayload = {
    agent_profile_id: options.agentProfileId,
    ...(Object.keys(launchAdditions).length > 0
      ? { agent_launch_additions: launchAdditions }
      : {}),
    workspace: conversationSettings.workspace,
    // The agent-server caches each client tool's schema per tool *name* for the
    // life of the process and rejects a re-registration with a different schema
    // (`ClientToolSchemaConflictError`). Editing either schema below therefore
    // requires restarting a long-running dev agent-server before new
    // conversations can start.
    client_tools:
      options.agentProfileKind === "openhands"
        ? [
            CANVAS_UI_CLIENT_TOOL,
            LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL,
            REQUEST_SECRET_CLIENT_TOOL,
            ...(options.automationSetup
              ? [AUTOMATION_FORM_UPDATE_CLIENT_TOOL]
              : []),
          ]
        : [],
    confirmation_policy:
      getConversationConfirmationPolicy(conversationSettings),
    max_iterations: resolveMaxIterations(conversationSettings.max_iterations),
    stuck_detection: true,
    autotitle: true,
    ...(options.titleLlmProfile
      ? { title_llm_profile: options.titleLlmProfile }
      : {}),
    worktree:
      options.executionRuntime === "docker"
        ? false
        : (options.worktree ?? true),
    ...getConversationRuntimeOptions(conversationSettings),
  };

  // Stamp the client source tag so the agent-server can attribute the
  // conversation to Canvas in telemetry (conversation_source = "canvas").
  payload.tags = {
    ...(options.automationSetup
      ? buildAutomationSetupModeTags(options.automationSetupTags)
      : {}),
    ...(options.agentProfileKind === "acp" && options.acpServer
      ? { [ACP_SERVER_TAG_KEY]: options.acpServer }
      : {}),
    [CLIENT_SOURCE_TAG_KEY]: AGENT_CANVAS_SOURCE,
  };

  if (options.conversationId) {
    payload.conversation_id = options.conversationId;
  }

  if (options.parentConversationId) {
    payload.parent_conversation_id = options.parentConversationId;
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
  } else if (options.workspaceHookConfig) {
    payload.hook_config = options.workspaceHookConfig;
  }

  const toolModuleQualnames = {
    ...((conversationSettings.tool_module_qualnames as
      | Record<string, string>
      | undefined) ?? {}),
  };
  delete toolModuleQualnames[LEGACY_CANVAS_UI_TOOL_NAME];
  delete toolModuleQualnames[CANVAS_UI_CLIENT_TOOL_NAME];
  delete toolModuleQualnames[LAUNCH_CHILD_CONVERSATION_TOOL_NAME];
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
  /** The skills the parent's agent profile selects. */
  skillEnablement?: SkillEnablement;
  /** Mirrors the main conversation's workspace selection — see buildConfiguredConversationSettings. */
  executionRuntime?: AgentServerInfo["execution_runtime"];
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
  // OpenHands app-server's PLANNING_AGENT_INSTRUCTION), preserving any saved
  // suffix.
  const agentContext = buildAgentContext(
    agentSettings,
    options.skillEnablement,
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
    workspace:
      options.executionRuntime === "docker"
        ? {
            kind: "DockerExecutionWorkspace",
            working_dir: DOCKER_EXECUTION_WORKING_DIR,
          }
        : {
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

/**
 * The skill selection of the parent's agent profile, or of the active one when
 * the parent recorded none. `null` — caller falls back to the global lists —
 * for an ACP profile or a failed lookup.
 */
async function resolveAgentProfileSkillEnablement(
  agentProfileId: string | null | undefined,
): Promise<SkillEnablement | null> {
  try {
    const { default: AgentProfilesService } =
      await import("./agent-profiles-service/agent-profiles-service.api");
    const { profiles, active_agent_profile_id: activeId } =
      await AgentProfilesService.listProfiles();
    const summary = profiles.find(
      (profile) => profile.id === (agentProfileId ?? activeId),
    );
    if (summary?.agent_kind !== "openhands") return null;
    const { profile } = await AgentProfilesService.getProfile(summary.name);
    return profile.agent_kind === "openhands"
      ? profileSkillEnablement(profile as StoredOpenHandsProfile)
      : null;
  } catch (error) {
    console.warn(
      "Falling back to global skill settings for the planner",
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
  /** The server's `execution_runtime` — threads through to workspace selection. */
  executionRuntime?: AgentServerInfo["execution_runtime"];
}): Promise<RawAgentStartConversationPayload> {
  const { SecretsService } = await import("./secrets-service");

  const [settingsResult, customSecrets, profileSkillSelection] =
    await Promise.all([
      SettingsService.getSettingsForConversation(),
      SecretsService.getSecrets(),
      resolveAgentProfileSkillEnablement(options.parentAgentProfileId),
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
    skillEnablement: profileSkillSelection ?? settingsResult.skillEnablement,
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
  automationSetup?: boolean;
  automationSetupTags?: Record<string, string>;
  conversationInstructions?: string;
  plugins?: PluginSpec[];
  conversationId?: string;
  parentConversationId?: string;
  workingDir?: string;
  /** Workspace root for the hooks lookup, not the per-conversation `workingDir` (#16907). */
  hooksProjectDir?: string | null;
  worktree?: boolean;
  agentProfileId: string;
  agentProfileKind?: AgentKind;
  llmProfileOverride?: string;
  profileHasRouter?: boolean;
  acpServer?: string;
  titleLlmProfile?: string;
}): Promise<Record<string, unknown>> {
  const [{ SecretsService }, { default: HooksService }] = await Promise.all([
    import("./secrets-service"),
    import("./hooks-service"),
  ]);

  const [settingsResult, customSecrets, serverInfo, workspaceHookConfig] =
    await Promise.all([
      SettingsService.getSettingsForConversation(),
      SecretsService.getSecrets(),
      fetchBackendServerInfo(),
      options.hooksProjectDir === null
        ? Promise.resolve(null)
        : HooksService.loadWorkspaceHooks(options.hooksProjectDir),
    ]);

  return buildStartConversationRequest({
    ...options,
    encryptedConversationSettings: settingsResult.conversationSettings,
    customSecrets,
    runtimeServicesInfo: parseRuntimeServicesInfo(
      (serverInfo as { runtime_services?: unknown } | null)?.runtime_services,
    ),
    executionRuntime: serverInfo?.execution_runtime,
    workspaceHookConfig,
  });
}

export function emptyHooksResponse(): GetHooksResponse {
  return { hooks: [] };
}

/** The hooks dialog's events: a workspace's configured events in SDK order. */
export function toHooksResponse(
  hookConfig: HookConfig | null,
): GetHooksResponse {
  if (!hookConfig) return emptyHooksResponse();
  return {
    hooks: [...HOOK_EVENT_FIELDS]
      .filter((eventType) => hookConfig[eventType]?.length)
      .map((eventType) => ({
        event_type: eventType,
        matchers: hookConfig[eventType].map(({ matcher, hooks }) => ({
          // The SDK's own defaults for fields a hooks.json may omit.
          matcher: matcher ?? "*",
          hooks: hooks.map((hook) => {
            // `prompt` and `system_prompt` reach Canvas over the wire but lag in
            // the client's HookDefinition; the server sends them as null when absent.
            const wireHook = hook as typeof hook & {
              prompt?: string | null;
              system_prompt?: string | null;
            };
            return {
              type: hook.type ?? HookType.COMMAND,
              command: hook.command,
              prompt: wireHook.prompt ?? undefined,
              system_prompt: wireHook.system_prompt ?? undefined,
              timeout: hook.timeout,
              async: hook.async,
            };
          }),
        })),
      })),
  };
}
