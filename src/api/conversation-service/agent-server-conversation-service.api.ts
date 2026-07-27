import {
  ConversationSortOrder,
  type ForkConversationRequest,
  type LLMConfig,
} from "@openhands/typescript-client";
import {
  ConversationClient,
  FileClient,
  ProfilesClient,
  VSCodeClient,
} from "@openhands/typescript-client/clients";
import { v4 as uuidv4 } from "uuid";
import { AgentKind, Provider } from "#/types/settings";
import type { ConversationRuntimeContext } from "#/api/conversation-file-upload.api";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import {
  buildConversationWorkingDir,
  getAgentServerWorkingDir,
} from "../agent-server-config";
import { resolveAbsoluteAgentServerPath } from "../agent-server-home";
import {
  getActiveBackend,
  getEffectiveLocalBackend,
} from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";
import ProfilesService from "../profiles-service/profiles-service.api";
import {
  batchGetCloudConversations,
  createCloudAppConversation,
  deleteCloudConversation,
  downloadCloudConversation,
  getCloudAppConversationStartTask,
  readCloudConversationFile,
  searchCloudConversations,
  updateCloudConversationPublicFlag,
} from "../cloud/conversation-service.api";
import {
  DirectConversationInfo,
  assertSubscriptionAuthReady,
  buildStartConversationRequestWithEncryptedSettings,
  emptyHooksResponse,
  getDefaultConversationTitle,
  toAppConversation,
  toConversationPage,
} from "../agent-server-adapter";
import { GetVSCodeUrlResponse } from "../open-hands.types";
import {
  getAgentServerClientOptions,
  NoBackendAvailableError,
} from "../agent-server-client-options";
import SettingsService from "../settings-service/settings-service.api";
import {
  ConversationMetadata,
  getStoredConversationMetadata,
  removeStoredConversationMetadata,
  setStoredConversationMetadata,
  type WorkspaceMode,
} from "../conversation-metadata-store";
import { resolveTitleLlmProfile } from "#/utils/title-llm-profile";
import type {
  GetHooksResponse,
  PluginSpec,
  AppConversation,
  AppConversationPage,
  AppConversationStartRequest,
  AppConversationStartTask,
  MetricsSnapshot,
  RuntimeConversationInfo,
  SendMessageRequest,
  SendMessageResponse,
} from "./agent-server-conversation-service.types";

const DEFAULT_CONVERSATION_TIMESTAMP = "1970-01-01T00:00:00.000Z";
// Creating the first conversation right after a cold agent-server boot (fresh
// machine or the packaged desktop app, where uvx may still be warming caches)
// can exceed the client's 60s default timeout.
const CREATE_CONVERSATION_TIMEOUT_MS = 5 * 60 * 1000;
const INVALID_CONVERSATION_RESPONSE_MESSAGE =
  "Unable to load conversations because the selected agent server returned " +
  "data this UI does not understand. Check the backend URL/session key and " +
  "update the agent server if needed.";
function invalidConversationResponse(): Error {
  return new Error(INVALID_CONVERSATION_RESPONSE_MESSAGE);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function numberOrZero(value: unknown): number {
  return typeof value === "number" ? value : 0;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readTimestamp(
  item: Record<string, unknown>,
  snakeKey: "created_at" | "updated_at",
  camelKey: "createdAt" | "updatedAt",
): string {
  const value = item[snakeKey] ?? item[camelKey];
  return typeof value === "string" && value.trim()
    ? value
    : DEFAULT_CONVERSATION_TIMESTAMP;
}

function normalizeTokenUsage(
  value: unknown,
): NonNullable<MetricsSnapshot["accumulated_token_usage"]> | null {
  if (!isRecord(value)) return null;

  return {
    prompt_tokens: numberOrZero(value.prompt_tokens),
    completion_tokens: numberOrZero(value.completion_tokens),
    cache_read_tokens: numberOrZero(value.cache_read_tokens),
    cache_write_tokens: numberOrZero(value.cache_write_tokens),
    context_window: numberOrZero(value.context_window),
    per_turn_token: numberOrZero(value.per_turn_token),
  };
}

function normalizeMetrics(value: unknown): MetricsSnapshot | null {
  if (!isRecord(value)) return null;

  return {
    accumulated_cost: numberOrNull(value.accumulated_cost),
    max_budget_per_task: numberOrNull(value.max_budget_per_task),
    accumulated_token_usage: normalizeTokenUsage(value.accumulated_token_usage),
  };
}

function normalizeAgent(value: unknown): DirectConversationInfo["agent"] {
  if (!isRecord(value)) return null;
  const llm = isRecord(value.llm)
    ? { model: stringOrNull(value.llm.model) }
    : null;
  // ``kind`` is the SDK's pydantic discriminator (``"Agent"`` vs ``"ACPAgent"``);
  // ``toAppConversation`` reads it to derive ``agent_kind``. ``acp_server`` is
  // the ACP provider identity (``ACPAgent.acp_server``, SDK #3692) — required so
  // the adapter can source the chip + in-conversation model list from the agent
  // when the ``acpserver`` tag is absent (a profile launch doesn't stamp it,
  // #1571). ``acp_model`` is the Canvas-configured model — preserved so the chip
  // can fall back to it when the SDK runtime model fields aren't populated.
  // Preserving these makes the wire path agree with the unit-test path that
  // builds ``DirectConversationInfo`` directly
  // (e.g. ``__tests__/api/agent-server-adapter.test.ts``).
  return {
    kind: stringOrNull(value.kind),
    acp_server: stringOrNull(value.acp_server),
    acp_model: stringOrNull(value.acp_model),
    llm,
  };
}

function normalizeWorkspace(
  value: unknown,
): DirectConversationInfo["workspace"] {
  if (!isRecord(value)) return null;
  return { working_dir: stringOrNull(value.working_dir) };
}

/**
 * Accept the agent-server's ``tags: Record[str, str]`` payload defensively:
 * the wire shape is guaranteed by the server-side validator (keys
 * ``^[a-z0-9]+$``, string values), but a non-conforming response (older
 * server, raw API write, future schema drift) must never crash the parser
 * — Canvas only consumes ``acpserver`` and falls back to a generic chip
 * for anything it doesn't recognize. Drop entries whose value isn't a
 * plain string; return ``null`` when the wire field is absent or not an
 * object so consumers can use ``info.tags?.[KEY] ?? null`` uniformly.
 */
function normalizeTags(value: unknown): Record<string, string> | null {
  if (!isRecord(value)) return null;
  const tags: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === "string") {
      tags[key] = entry;
    }
  }
  return tags;
}

function normalizeLaunchedAgentProfile(
  value: unknown,
): DirectConversationInfo["launched_agent_profile"] {
  if (!isRecord(value)) return null;
  const { agent_profile_id: agentProfileId, revision } = value;
  if (typeof agentProfileId !== "string" || typeof revision !== "number") {
    return null;
  }
  return { agent_profile_id: agentProfileId, revision };
}

function normalizeAbsolutePath(path: string): string | null {
  if (!path.startsWith("/")) return null;

  const segments: string[] = [];
  for (const segment of path.split("/")) {
    if (segment && segment !== ".") {
      if (segment === "..") {
        if (!segments.length) return null;
        segments.pop();
      } else {
        segments.push(segment);
      }
    }
  }

  return `/${segments.join("/")}`;
}

function requirePathInsideDirectory(path: string, directory: string): string {
  const normalizedPath = normalizeAbsolutePath(path);
  const normalizedDirectory = normalizeAbsolutePath(directory);

  if (
    !normalizedPath ||
    !normalizedDirectory ||
    (normalizedPath !== normalizedDirectory &&
      !normalizedPath.startsWith(`${normalizedDirectory}/`))
  ) {
    throw new Error("Conversation file path must stay inside the workspace");
  }

  return normalizedPath;
}

function requireDirectConversationInfo(item: unknown): DirectConversationInfo {
  if (!isRecord(item) || typeof item.id !== "string" || !item.id.trim()) {
    throw invalidConversationResponse();
  }

  return {
    id: item.id.trim(),
    title: stringOrNull(item.title),
    created_at: readTimestamp(item, "created_at", "createdAt"),
    updated_at: readTimestamp(item, "updated_at", "updatedAt"),
    execution_status: stringOrNull(item.execution_status),
    sandbox_status: stringOrNull(item.sandbox_status),
    metrics: normalizeMetrics(item.metrics),
    agent: normalizeAgent(item.agent),
    workspace: normalizeWorkspace(item.workspace),
    tags: normalizeTags(item.tags),
    launched_agent_profile: normalizeLaunchedAgentProfile(
      item.launched_agent_profile,
    ),
    // SDK-runtime ACP model fields (populated when the agent-server supports
    // ``ConversationInfo.current_model_*``). Consumed by the conversation
    // adapter to drive the per-card chip's model text. Older agent-servers
    // omit these — adapter handles ``undefined`` / ``null`` gracefully.
    current_model_id: stringOrNull(item.current_model_id),
    current_model_name: stringOrNull(item.current_model_name),
  };
}

function requireDirectConversationItems(
  items: unknown,
): DirectConversationInfo[] {
  if (!Array.isArray(items)) {
    throw invalidConversationResponse();
  }
  return items.map(requireDirectConversationInfo);
}

function requireConversationSearchPage(page: unknown): {
  items: DirectConversationInfo[];
  next_page_id: string | null;
} {
  if (Array.isArray(page)) {
    return {
      items: requireDirectConversationItems(page),
      next_page_id: null,
    };
  }

  if (!isRecord(page)) {
    throw invalidConversationResponse();
  }

  return {
    items: requireDirectConversationItems(page.items),
    next_page_id:
      typeof page.next_page_id === "string" ? page.next_page_id : null,
  };
}

const RUNTIME_STATUSES = new Set<string>([
  "idle",
  "running",
  "paused",
  "waiting_for_confirmation",
  "finished",
  "error",
  "stuck",
]);

function toRuntimeStatus(
  status: DirectConversationInfo["execution_status"],
): RuntimeConversationInfo["status"] {
  const nextStatus = status ?? "idle";
  return (
    RUNTIME_STATUSES.has(nextStatus) ? nextStatus : "idle"
  ) as RuntimeConversationInfo["status"];
}

function requireAppConversation(
  conversation: AppConversation | null | undefined,
  conversationId: string,
): AppConversation {
  if (!conversation) {
    throw new Error(`Conversation ${conversationId} was not found`);
  }
  return conversation;
}

class AgentServerConversationService {
  static async sendMessage(
    conversationId: string,
    message: SendMessageRequest,
    runtime?: ConversationRuntimeContext | null,
  ): Promise<SendMessageResponse> {
    const active = getActiveBackend().backend;
    let conversationUrl = runtime?.conversationUrl ?? null;
    let sessionApiKey = runtime?.sessionApiKey ?? null;

    if (active.kind === "cloud") {
      if (!conversationUrl || !sessionApiKey) {
        const [conversation] = await batchGetCloudConversations([
          conversationId,
        ]);
        conversationUrl = conversation?.conversation_url?.trim() ?? null;
        sessionApiKey = conversation?.session_api_key?.trim() ?? null;
      }

      if (!conversationUrl || !sessionApiKey) {
        throw new Error(
          "Conversation sandbox is still starting. Wait for it to finish, then try again.",
        );
      }

      await callCloudProxy({
        backend: active,
        method: "POST",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: `/api/conversations/${conversationId}/events`,
        body: { ...message, run: true },
        authMode: "session-api-key",
        sessionApiKey,
      });

      return message;
    }

    await new ConversationClient(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).sendEvent(conversationId, message, {
      run: true,
    });

    return message;
  }

  static async createConversation(
    initialUserMsg?: string,
    conversationInstructions?: string,
    plugins?: PluginSpec[],
    metadata?: ConversationMetadata | null,
    workingDirOverride?: string,
    workspaceMode?: WorkspaceMode,
    parentConversationId?: string,
    agentType?: "default" | "plan",
    sandboxId?: string,
    // Launch from a saved AgentProfile (resolved server-side) instead of the
    // current encrypted agent_settings (#3727). Supported on both local and the
    // cloud app-server (OpenHands #15060): local threads it through the
    // encrypted-settings builder; cloud sends it as a flat request field.
    agentProfileId?: string,
    agentProfileKind?: AgentKind,
  ): Promise<AppConversationStartTask> {
    if (getActiveBackend().backend.kind === "cloud") {
      // Cloud path mirrors OpenHands' frontend: build a flat
      // AppConversationStartRequest, POST /api/v1/app-conversations
      // (returns a WORKING task), and let the conversation route's
      // useTaskPolling drive it to READY. NO encrypted-settings
      // round-trip — the cloud backend holds secrets server-side.
      // When launching from a profile, send `agent_profile_id`; the backend
      // resolves it to agent_settings server-side.
      const request: AppConversationStartRequest = {
        initial_message: initialUserMsg
          ? {
              role: "user",
              content: [{ type: "text", text: initialUserMsg }],
            }
          : null,
        title: conversationInstructions ?? null,
        selected_repository: metadata?.selected_repository ?? null,
        selected_branch: metadata?.selected_branch ?? null,
        git_provider: metadata?.git_provider ?? null,
        plugins: plugins ?? null,
        parent_conversation_id: parentConversationId ?? null,
        agent_type: agentType,
        sandbox_id: sandboxId ?? null,
        agent_profile_id: agentProfileId ?? null,
      };
      return createCloudAppConversation(request);
    }

    const [settings, profiles] = await Promise.all([
      SettingsService.getSettings(),
      ProfilesService.listProfiles().catch(() => undefined),
    ]);
    const titleLlmProfile = resolveTitleLlmProfile(
      settings.title_llm_profile,
      profiles,
    );
    const conversationId = uuidv4();
    // @spec WUP-001 — Send an absolute working_dir to the agent-server.
    // The default is `workspace/project/<hex>` (relative); without
    // resolving it here, `/api/file/upload` later prepends `/` and writes
    // to `/workspace/...` (read-only on macOS and fresh containers). When
    // the user picks an explicit workspace, `workingDirOverride` is
    // already absolute (it comes from `search_subdirs`).
    const workingDir = await resolveAbsoluteAgentServerPath(
      workingDirOverride ?? buildConversationWorkingDir(conversationId),
    );
    const resolvedWorkspaceMode =
      workspaceMode ?? (workingDirOverride ? "local_repo" : "new_worktree");

    // Use encrypted settings to avoid exposing secrets in the browser
    const payload = await buildStartConversationRequestWithEncryptedSettings({
      settings,
      query: initialUserMsg,
      conversationInstructions,
      plugins,
      conversationId,
      workingDir,
      worktree: resolvedWorkspaceMode === "new_worktree",
      agentProfileId,
      agentProfileKind,
      titleLlmProfile,
    });

    const data = await new ConversationClient(
      getAgentServerClientOptions({ timeout: CREATE_CONVERSATION_TIMEOUT_MS }),
    ).createConversation<DirectConversationInfo>(payload);
    const localBackend = getEffectiveLocalBackend();
    if (!localBackend) throw new NoBackendAvailableError();

    if (metadata?.selected_repository || workingDirOverride) {
      // The agent-server runtime has no concept of selected repo/branch/
      // workspace, so persist the home-page selection client-side.
      // `toAppConversation` reads the repo/branch fields back to hydrate
      // the chat-page badges; `useHasAttachedSource` reads
      // `selected_workspace` to default the Files tab to Diff mode when
      // the user explicitly attached a local workspace.
      setStoredConversationMetadata(data.id, {
        selected_repository: metadata?.selected_repository ?? null,
        selected_branch: metadata?.selected_branch ?? null,
        git_provider: metadata?.git_provider ?? null,
        selected_workspace: workingDirOverride ?? null,
        workspace_mode: resolvedWorkspaceMode,
      });
    }

    return {
      id: data.id,
      created_by_user_id: null,
      status: "READY",
      detail: null,
      app_conversation_id: data.id,
      agent_server_url: localBackend.host,
      request: {
        initial_message: payload.initial_message as
          | AppConversationStartRequest["initial_message"]
          | undefined,
        plugins: plugins ?? null,
      },
      created_at: data.created_at,
      updated_at: data.updated_at,
    };
  }

  static async getStartTask(
    taskId: string,
  ): Promise<AppConversationStartTask | null> {
    if (getActiveBackend().backend.kind === "cloud") {
      return getCloudAppConversationStartTask(taskId);
    }
    // Local agent-server creates conversations synchronously — every
    // local "task" is already READY when createConversation returns, so
    // there's nothing to poll for.
    return null;
  }

  static async getVSCodeUrl(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey?: string | null,
  ): Promise<GetVSCodeUrlResponse> {
    // Local-only path. Cloud conversations read the VSCode URL straight
    // from the cloud-computed `sandbox.exposed_urls` (see
    // `useUnifiedVSCodeUrl` + `useCloudSandbox`); the runtime's own
    // `/api/vscode/url` only knows its internal `localhost:8001`, which
    // the user's browser can't reach.
    const workspaceDir =
      await this.resolveConversationWorkingDir(conversationId);
    // Local mode: the typescript-client targets the local agent-server
    // directly via the conversationUrl override.
    const vscodeUrl = await new VSCodeClient(
      getAgentServerClientOptions({
        conversationUrl,
        sessionApiKey,
      }),
    ).getUrl({
      baseUrl:
        typeof window !== "undefined" ? window.location.origin : undefined,
      workspaceDir,
    });

    return { vscode_url: vscodeUrl };
  }

  static async resolveConversationWorkingDir(
    conversationId: string,
  ): Promise<string> {
    const [conversation] = await this.batchGetAppConversations([
      conversationId,
    ]);
    return conversation?.workspace?.working_dir ?? getAgentServerWorkingDir();
  }

  static async batchGetAppConversations(
    ids: string[],
  ): Promise<(AppConversation | null)[]> {
    if (ids.length === 0) return [];

    if (getActiveBackend().backend.kind === "cloud") {
      return batchGetCloudConversations(ids);
    }

    const data = await new ConversationClient(
      getAgentServerClientOptions(),
    ).getConversations<DirectConversationInfo>(ids);

    return requireDirectConversationItems(data).map((item) =>
      toAppConversation(item),
    );
  }

  static async updateConversationPublicFlag(
    conversationId: string,
    isPublic: boolean,
  ): Promise<AppConversation> {
    if (getActiveBackend().backend.kind !== "cloud") {
      throw new Error("Public sharing requires a cloud backend.");
    }
    return updateCloudConversationPublicFlag(conversationId, isPublic);
  }

  static async updateConversationRepository(
    conversationId: string,
    repository: string | null,
    branch?: string | null,
    gitProvider?: string | null,
  ): Promise<AppConversation> {
    if (repository) {
      const existing = getStoredConversationMetadata(conversationId);
      setStoredConversationMetadata(conversationId, {
        ...(existing ?? {}),
        selected_repository: repository,
        selected_branch: branch ?? null,
        git_provider: (gitProvider as Provider | null | undefined) ?? null,
      });
    } else {
      removeStoredConversationMetadata(conversationId);
    }
    const [conversation] = await this.batchGetAppConversations([
      conversationId,
    ]);
    return requireAppConversation(conversation, conversationId);
  }

  static async readConversationFile(
    conversationId: string,
    filePath?: string,
  ): Promise<string> {
    if (getActiveBackend().backend.kind === "cloud") {
      // Cloud exposes a per-conversation file endpoint; the sandbox
      // working dir is fixed (`/workspace/project`), so PLAN.md lives at
      // a known absolute path. Mirrors OpenHands' readConversationFile.
      const path = requirePathInsideDirectory(
        filePath ?? "/workspace/project/.agents_tmp/PLAN.md",
        "/workspace/project",
      );
      return readCloudConversationFile(conversationId, path);
    }

    const workingDir = await this.resolveConversationWorkingDir(conversationId);
    const path = requirePathInsideDirectory(
      filePath ?? `${workingDir}/.agents_tmp/PLAN.md`,
      workingDir,
    );
    return new FileClient(getAgentServerClientOptions()).downloadTextFile(path);
  }

  static async downloadConversation(conversationId: string): Promise<Blob> {
    if (getActiveBackend().backend.kind === "cloud") {
      return downloadCloudConversation(conversationId);
    }

    return new FileClient(getAgentServerClientOptions()).downloadTrajectory(
      conversationId,
    );
  }

  static async getHooks(conversationId: string): Promise<GetHooksResponse> {
    if (!conversationId) {
      return emptyHooksResponse();
    }
    return emptyHooksResponse();
  }

  static async getRuntimeConversation(
    conversationId: string,
    conversationUrl: string | null | undefined,
    sessionApiKey?: string | null,
  ): Promise<RuntimeConversationInfo> {
    const active = getActiveBackend().backend;

    type RawRuntime = DirectConversationInfo & {
      stats?: RuntimeConversationInfo["stats"];
    };

    // Cloud mode: route through the cloud-proxy to the runtime sandbox at
    // the conversation's runtime URL — same pattern as getVSCodeUrl. Local
    // mode forwards conversationUrl so the host explicitly resolves to the
    // conversation's runtime instead of falling back to the active backend.
    const response =
      active.kind === "cloud" && conversationUrl
        ? await callCloudProxy<RawRuntime>({
            backend: active,
            method: "GET",
            hostOverride: buildHttpBaseUrl(conversationUrl),
            path: `/api/conversations/${conversationId}`,
            authMode: "session-api-key",
            sessionApiKey,
          })
        : await new ConversationClient(
            getAgentServerClientOptions({
              conversationUrl,
              sessionApiKey,
            }),
          ).getConversation<RawRuntime>(conversationId);
    const data = requireDirectConversationInfo(response);
    const stats = isRecord(response) ? response.stats : null;

    return {
      id: data.id,
      title: data.title?.trim()
        ? data.title
        : getDefaultConversationTitle(data.id),
      metrics: normalizeMetrics(data.metrics),
      created_at: data.created_at,
      updated_at: data.updated_at,
      status: toRuntimeStatus(data.execution_status),
      stats: isRecord(stats) ? stats : { usage_to_metrics: {} },
    };
  }

  static async searchConversations(
    limit: number = 20,
    pageId?: string,
  ): Promise<AppConversationPage> {
    if (getActiveBackend().backend.kind === "cloud") {
      return searchCloudConversations(limit, pageId);
    }

    const data = await new ConversationClient(
      getAgentServerClientOptions(),
    ).searchConversations({
      limit,
      page_id: pageId,
      sort_order: ConversationSortOrder.UPDATED_AT_DESC,
    });

    return toConversationPage(requireConversationSearchPage(data));
  }

  static async deleteConversation(conversationId: string): Promise<void> {
    if (getActiveBackend().backend.kind === "cloud") {
      await deleteCloudConversation(conversationId);
    } else {
      await new ConversationClient(
        getAgentServerClientOptions(),
      ).deleteConversation(conversationId);
    }
    removeStoredConversationMetadata(conversationId);
  }

  static async updateConversationTitle(
    conversationId: string,
    title: string,
  ): Promise<AppConversation> {
    await new ConversationClient(
      getAgentServerClientOptions(),
    ).updateConversation(conversationId, {
      title,
    });
    const [conversation] = await this.batchGetAppConversations([
      conversationId,
    ]);
    return requireAppConversation(conversation, conversationId);
  }

  /**
   * Forks a conversation, copying event history up to and including
   * `fromEventId`. Local agent-server only; needs agent-server >= 1.31.0 for
   * `from_event_id` (older backends copy the whole conversation).
   */
  static async forkConversation(
    sourceConversationId: string,
    fromEventId: string,
    title?: string,
  ): Promise<DirectConversationInfo> {
    if (getActiveBackend().backend.kind === "cloud") {
      throw new Error(
        "Branching a conversation isn't supported on the cloud backend yet.",
      );
    }

    // `from_event_id` is accepted by `/fork` but not yet typed in
    // ForkConversationRequest (through client 1.32.0); the client forwards the
    // body verbatim, so cast to carry it. A title also suppresses the backend
    // auto-title, so the "(branch)" marker sticks.
    const data = await new ConversationClient(
      getAgentServerClientOptions(),
    ).forkConversation<DirectConversationInfo>(sourceConversationId, {
      from_event_id: fromEventId,
      ...(title ? { title } : {}),
    } as ForkConversationRequest & { from_event_id: string });

    // Carry over the source's client-side metadata (repo/branch/workspace/
    // profile/plugins) so the fork hydrates its chat-page badges the same way.
    const sourceMetadata = getStoredConversationMetadata(sourceConversationId);
    if (sourceMetadata) {
      setStoredConversationMetadata(data.id, sourceMetadata);
    }

    return data;
  }

  /**
   * Returns an event's `parent_id` (the fork point for branching *before* it),
   * or undefined at the root. Uses the single-event endpoint because the events
   * *search* API omits `parent_id`.
   */
  static async getEventParentId(
    conversationId: string,
    eventId: string,
  ): Promise<string | undefined> {
    const event = (await new ConversationClient(
      getAgentServerClientOptions(),
    ).getEvent(conversationId, eventId)) as { parent_id?: string | null };
    return event.parent_id ?? undefined;
  }

  /**
   * Switches the LLM profile for the running conversation when one is open
   * (POST /switch_profile — per-conversation swap, doesn't change the user's
   * default profile). When called without a conversationId (home page),
   * falls back to POST /activate so the next conversation created picks up
   * the chosen profile.
   *
   * The per-conversation endpoint accepts only the profile name, so the UI does
   * not need to fetch or forward profile secrets. That keeps switching working
   * even when the agent server has no OH_SECRET_KEY for encrypted secret export.
   *
   * Cloud backends route to the app-server's per-conversation
   * `/switch_profile`, which owns the profiles and resolves the swap
   * server-side (base_url/api_key fixups, usage_id derivation, then the
   * agent-server's switch_llm) — so the client only forwards the profile name,
   * mirroring {@link switchAcpModel}.
   */
  static async switchProfile(
    conversationId: string | null,
    profileName: string,
  ): Promise<void> {
    const { backend } = getActiveBackend();

    if (backend.kind === "cloud") {
      // No conversation (home page): activate globally so the next
      // conversation starts with it. ProfilesService routes to the cloud
      // activate endpoint.
      if (!conversationId) {
        await ProfilesService.activateProfile(profileName);
        return;
      }
      await callCloudProxy({
        backend,
        method: "POST",
        path: `/api/v1/app-conversations/${conversationId}/switch_profile`,
        body: { profile_name: profileName },
      });
      return;
    }

    if (!conversationId) {
      await new ProfilesClient(getAgentServerClientOptions()).activateProfile(
        profileName,
      );
      return;
    }

    const clientOptions = getAgentServerClientOptions();
    const conversationClient = new ConversationClient(clientOptions);
    const profile = await new ProfilesClient(clientOptions).getProfile(
      profileName,
      { exposeSecrets: "encrypted" },
    );
    const model =
      typeof profile.config.model === "string" ? profile.config.model : "";
    if (!model) throw new Error(`Profile '${profileName}' has no model.`);
    await assertSubscriptionAuthReady({ llm: profile.config });
    await conversationClient.switchLLM(conversationId, {
      ...profile.config,
      model,
      // Keep streaming on after a switch (parity with conversation start);
      // the profile config would otherwise default it to stream=False.
      stream: true,
      // Avoid stale first-write-wins entries in the backend LLM registry.
      usage_id: `profile:${profileName}:${uuidv4()}`,
    } as LLMConfig);
  }

  /**
   * Switches the model of a running ACP conversation in place (POST
   * /switch_acp_model — the ACP analog of {@link switchProfile}'s /switch_profile).
   * The agent-server calls the ACP wrapper's ``session/set_model`` on the live
   * session, preserving context. Mirrors {@link switchProfile}'s
   * local-backend-only guard and per-conversation ConversationClient call.
   *
   * Only valid once an ACP session exists (after the first message); the
   * agent-server returns 409 before then — the home/no-session default is
   * persisted via Settings instead (see ``use-switch-acp-model``).
   */
  static async switchAcpModel(
    conversationId: string,
    model: string,
  ): Promise<void> {
    const { backend } = getActiveBackend();
    if (backend.kind === "cloud") {
      await callCloudProxy({
        backend,
        method: "POST",
        path: `/api/v1/app-conversations/${conversationId}/switch_acp_model`,
        body: { model },
      });
      return;
    }

    await new ConversationClient(getAgentServerClientOptions()).switchAcpModel(
      conversationId,
      model,
    );
  }
}

export default AgentServerConversationService;
