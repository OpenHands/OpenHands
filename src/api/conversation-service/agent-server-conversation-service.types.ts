import { ConversationTrigger } from "../open-hands.types";
import { Provider } from "#/types/settings";
import { SuggestedTask } from "#/utils/types";
import { ExecutionStatus } from "#/types/agent-server/core";

/**
 * Lifecycle state of a cloud sandbox. Mirrors OpenHands' V1SandboxStatus.
 * Local agent-server conversations do not carry this field (null).
 */
export type SandboxStatus =
  | "PAUSED"
  | "RUNNING"
  | "STARTING"
  | "MISSING"
  | "ERROR";

// Plugin specification for starting conversations with plugins
export interface PluginSpec {
  source: string; // Plugin source: 'github:owner/repo', git URL, or local path
  ref?: string | null; // Optional branch, tag, or commit
  repo_path?: string | null; // Subdirectory path within the git repository
  parameters?: Record<string, unknown> | null; // User-provided configuration values
  // Client-side display label (e.g. an installed plugin's `name`). Not a
  // coordinate and not required by the agent-server; used only for rendering.
  name?: string | null;
}

// V1 Metrics Types
export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  context_window: number;
  per_turn_token: number;
}

export interface MetricsSnapshot {
  accumulated_cost: number | null;
  max_budget_per_task: number | null;
  accumulated_token_usage: TokenUsage | null;
}

// V1 API Types for requests
// These types match the SDK's TextContent and ImageContent formats
export interface MessageTextContent {
  type: "text";
  text: string;
}

export interface MessageImageContent {
  type: "image";
  image_urls: string[];
}

export type MessageContent = MessageTextContent | MessageImageContent;

type MessageRole = "user" | "system" | "assistant" | "tool";

export interface SendMessageRequest {
  role: MessageRole;
  content: MessageContent[];
}

export interface AppConversationStartRequest {
  // Re-provision an EXISTING conversation (waking a recycled sandbox) instead
  // of minting a new one. The backend keys the rebuilt conversation on this id
  // and, for ACP, resumes it from the durable event store with a bootstrap
  // prompt (OpenHands#14640). Omit/null to create a fresh conversation.
  conversation_id?: string | null;
  initial_message?: SendMessageRequest | null;
  processors?: unknown[]; // EventCallbackProcessor - keeping as unknown for now
  llm_model?: string | null;
  selected_repository?: string | null;
  selected_branch?: string | null;
  git_provider?: Provider | null;
  suggested_task?: SuggestedTask | null;
  title?: string | null;
  trigger?: ConversationTrigger | null;
  pr_number?: number[];
  parent_conversation_id?: string | null;
  agent_type?: "default" | "plan";
  sandbox_id?: string | null;
  plugins?: PluginSpec[] | null; // Plugins to load when starting the conversation
  // One-off launch from a saved AgentProfile, resolved server-side (#3727).
  // Accepted by the cloud app-server (OpenHands #15060) on
  // POST /api/v1/app-conversations; the local path uses the encrypted
  // agent_settings builder instead, which threads its own agentProfileId.
  agent_profile_id?: string | null;
}

export type AppConversationStartTaskStatus =
  | "WORKING"
  | "WAITING_FOR_SANDBOX"
  | "PREPARING_REPOSITORY"
  | "RUNNING_SETUP_SCRIPT"
  | "SETTING_UP_GIT_HOOKS"
  | "SETTING_UP_SKILLS"
  | "STARTING_CONVERSATION"
  | "READY"
  | "ERROR";

export interface AppConversationStartTask {
  id: string;
  created_by_user_id: string | null;
  status: AppConversationStartTaskStatus;
  detail: string | null;
  app_conversation_id: string | null;
  agent_server_url: string | null;
  request: AppConversationStartRequest;
  created_at: string;
  updated_at: string;
}

export interface SendMessageResponse {
  role: "user" | "system" | "assistant" | "tool";
  content: MessageContent[];
}

export interface AppConversationStartTaskPage {
  items: AppConversationStartTask[];
  next_page_id: string | null;
}

export interface ConversationWorkspace {
  working_dir: string | null;
}

export interface LaunchedAgentProfile {
  agent_profile_id: string;
  revision: number;
}

export interface AppConversation {
  id: string;
  created_by_user_id: string | null;
  selected_repository: string | null;
  selected_branch: string | null;
  git_provider: Provider | null;
  title: string | null;
  trigger: ConversationTrigger | null;
  pr_number: number[];
  /**
   * High-level kind of the conversation's agent — ``"openhands"`` for an LLM-
   * driven Agent, ``"acp"`` for an ACPAgent that delegates to an external
   * ACP CLI subprocess. Consumers can use this to gate UI affordances that
   * only make sense for one kind (e.g. the LLM-profile switcher in the chat
   * header is a no-op for ACP conversations even though ``llm_model`` may
   * carry the ACP subprocess model for display).
   */
  agent_kind?: "openhands" | "acp" | null;
  launched_agent_profile?: LaunchedAgentProfile | null;
  /**
   * For ACP conversations, the registry key of the ACP CLI server the
   * conversation was launched against (e.g. ``"claude-code"``, ``"codex"``,
   * ``"gemini-cli"``). Populated from ``info.tags.acpserver`` — see
   * ``ACP_SERVER_TAG_KEY`` in ``agent-server-adapter.ts`` for the wire
   * format and the rationale behind the snake_case-incompatible
   * ``acpserver`` form. ``null`` for OpenHands conversations and for ACP
   * conversations whose tag wasn't stamped (e.g. created via an older
   * client or via the raw API). Consumers resolve the display name via
   * ``getAcpProviderDisplayName(acp_server)`` and fall back to a generic
   * "ACP" chip when the key is unknown or null.
   */
  acp_server?: string | null;
  /**
   * Server-side key-value tags from the agent-server's
   * ``ConversationInfo.tags`` (settable at creation and via
   * ``PATCH /api/conversations/{id}``). Automations and API clients use these
   * for attribution (e.g. ``origin=slack``, ``owner=alice``). Includes
   * reserved/internal keys such as ``acpserver`` — display consumers should
   * go through ``getDisplayConversationTags`` in ``agent-server-adapter.ts``
   * rather than reading this raw. ``null``/absent for Cloud conversations
   * and agent-servers that don't return tags.
   *
   * PATCH semantics: the agent-server replaces the **complete** ``tags`` map
   * on every PATCH — it does not merge individual keys. A writer adding one
   * tag must read the current map, merge, and send the full desired result,
   * or it will silently drop every other tag (including ``acpserver``).
   */
  tags?: Record<string, string> | null;
  llm_model: string | null;
  metrics: MetricsSnapshot | null;
  created_at: string;
  updated_at: string;
  execution_status: ExecutionStatus | null;
  /**
   * Cloud-only sandbox lifecycle status. Mirrors OpenHands' V1SandboxStatus.
   * Absent / null for local agent-server conversations.
   */
  sandbox_status?: SandboxStatus | null;
  conversation_url: string | null;
  session_api_key: string | null;
  sandbox_id: string | null;
  workspace?: ConversationWorkspace | null;
  /**
   * The local workspace the user explicitly attached when creating this
   * conversation. Client-side only — never round-tripped to the agent-server
   * or cloud. Null/undefined for conversations created via "No workspace".
   * Distinct from `workspace.working_dir`, which is the runtime path and may
   * either match this folder directly or point at a per-conversation worktree.
   */
  selected_workspace?: string | null;
  /**
   * The LLM profile this conversation was created with / last switched to.
   * Hydrated from client-side metadata (see
   * `ConversationMetadata.active_profile`). Preferred over matching
   * `llm_model` against the profile list, which is ambiguous when several
   * profiles share a model (#1082). Null when unknown (e.g. created by an
   * older client) — consumers fall back to model-matching.
   */
  active_profile?: string | null;
  public?: boolean;
  sub_conversation_ids: string[];
}

export interface AppConversationPage {
  items: AppConversation[];
  next_page_id: string | null;
}

export interface Skill {
  name: string;
  type: "repo" | "knowledge" | "agentskills";
  content: string;
  triggers: string[];
  source?: string | null;
  description?: string | null;
  version?: string;
  license?: string | null;
  compatibility?: string | null;
  metadata?: Record<string, string> | null;
  allowed_tools?: string[] | null;
  is_agentskills_format?: boolean;
  disable_model_invocation?: boolean;
}

export interface GetSkillsResponse {
  skills: Skill[];
}

export interface HookDefinition {
  type: string; // 'command' or 'prompt'
  command: string;
  timeout: number;
  async?: boolean;
}

export interface HookMatcher {
  matcher: string; // Pattern: '*', exact match, or regex
  hooks?: HookDefinition[]; // May be undefined while hooks are still executing on the server
}

export interface HookEvent {
  event_type: string; // e.g., 'stop', 'pre_tool_use', 'post_tool_use'
  matchers: HookMatcher[];
}

export interface GetHooksResponse {
  hooks: HookEvent[];
}

// Runtime conversation types (from agent server)
export interface RuntimeConversationStats {
  usage_to_metrics: Record<string, RuntimeMetrics>;
}

export interface RuntimeMetrics {
  model_name: string;
  accumulated_cost: number;
  max_budget_per_task: number | null;
  accumulated_token_usage: TokenUsage | null;
  costs: Cost[];
  response_latencies: ResponseLatency[];
  token_usages: TokenUsage[];
}

export interface Cost {
  model: string;
  cost: number;
  timestamp: number;
}

export interface ResponseLatency {
  model: string;
  latency: number;
  response_id: string;
}

export interface RuntimeConversationInfo {
  id: string;
  title: string | null;
  metrics: MetricsSnapshot | null;
  created_at: string;
  updated_at: string;
  status: ExecutionStatus;
  stats: RuntimeConversationStats;
}
