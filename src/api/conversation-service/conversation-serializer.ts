/**
 * Conversation serialization: the agent-server's `DirectConversationInfo` wire
 * shape adapted into the `AppConversation` the UI consumes.
 *
 * Split out of `agent-server-adapter.ts` as the narrowest seam in that module —
 * it is pure adaptation with no request-construction or settings concerns, and
 * it already had focused tests. Moved verbatim; no behaviour change.
 */

import { DEFAULT_SETTINGS } from "#/services/settings";
import { ExecutionStatus } from "#/types/agent-server/core";
import { resolveEffectiveAcpModel } from "#/constants/acp-providers";
import { getAgentServerClientOptions } from "../agent-server-client-options";
import { getAgentServerWorkingDir } from "../agent-server-config";
import { getStoredConversationMetadata } from "../conversation-metadata-store";
import { ACP_SERVER_TAG_KEY } from "./conversation-tag-keys";
import {
  AppConversation,
  AppConversationPage,
  SandboxStatus,
} from "./agent-server-conversation-service.types";

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
    sub_conversation_ids: [],
  };
}

export function toConversationPage(data: {
  items: DirectConversationInfo[];
  next_page_id?: string | null;
}): AppConversationPage {
  return {
    items: data.items.map(toAppConversation),
    next_page_id: data.next_page_id ?? null,
  };
}
