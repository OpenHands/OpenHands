import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { QueryClient } from "@tanstack/react-query";
import {
  PluginSpec,
  MessageContent,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import { SuggestedTask } from "#/utils/types";
import { Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useSettings } from "#/hooks/query/use-settings";
import { useMetaProfiles } from "#/hooks/query/use-meta-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import PluginsManagementService, {
  type InstalledPluginInfo,
} from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { launchConversation } from "#/hooks/mutation/resolve-conversation-launch";
import {
  ConversationFirstMessageError,
  isDefinitiveSendRejection,
} from "#/api/conversation-service/agent-server-conversation-service.api";
import type { SetupScriptStatusUpdate } from "#/api/setup-script-service";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { pluginReferenceKey } from "#/utils/plugin-display";
import {
  mergeStoredConversationMetadata,
  toPluginCoordinates,
  type WorkspaceMode,
} from "#/api/conversation-metadata-store";

export interface CreateConversationVariables {
  query?: string;
  automationSetup?: boolean;
  automationSetupTags?: Record<string, string>;
  repository?: {
    name: string;
    gitProvider: Provider;
    branch?: string;
  };
  suggestedTask?: SuggestedTask;
  conversationInstructions?: string;
  parentConversationId?: string;
  agentType?: "default" | "plan";
  plugins?: PluginSpec[];
  workingDir?: string;
  workspaceMode?: WorkspaceMode;
  // Launch from a specific AgentProfile (local backend). When omitted, the
  // active AgentProfile (if any) is used so home-composed conversations
  // launch from the user's selected profile (#3727).
  agentProfileId?: string;
  // Cloud: run on this LLM profile (a member's home pick) without activating it.
  llmProfile?: string;
  entryPoint?: string; // analytics only; not forwarded to the service
  /**
   * Progress of the workspace bootstrap (`.openhands/setup.sh`), so the caller
   * can show an indicator while it runs. Not forwarded to the service as data:
   * the callback itself is passed straight through.
   */
  onSetupScriptProgress?: (update: SetupScriptStatusUpdate) => void;
}

export const CREATE_CONVERSATION_MUTATION_KEY = ["create-conversation"];

interface CreateConversationResponse {
  conversation_id: string;
  session_api_key: string | null;
  url: string | null;
  task_id?: string;
}

/**
 * Flatten a message-content array to the exact text the server will echo
 * back (the client joins `TextContent` parts; image parts carry no text).
 * Matches `extractMessageEventText` in the websocket context, which is what
 * consumes the optimistic queue.
 */
function messageContentText(content: MessageContent[]): string {
  return content
    .filter((part): part is MessageContent & { type: "text" } => {
      return part.type === "text";
    })
    .map((part) => part.text)
    .join("");
}

/**
 * The conversation was created, but its withheld first message never sent, so
 * its first turn never started. Reporting a failed create here would be worse
 * than useless: the caller's retry starts a *second* conversation and leaves
 * this one idle. Instead report success for the conversation that does exist
 * and leave the user's message as a pending bubble on it, so the retry
 * affordance resends to the same conversation.
 *
 * The bubble is rebuilt from the exact `content` the send attempted, not the
 * caller's raw `query`: the local path folds `conversationInstructions` into
 * the body, so a `query`-only rebuild would drop them (and produce no bubble
 * at all for an instructions-only launch).
 *
 * Only a *definitive* rejection (the server answered with a non-2xx) becomes an
 * errored bubble with retry. An ambiguous failure — the request may have been
 * accepted before the response was lost — stays "sending": the server's echo
 * removes it via `consumeMatchingPendingMessage`, and the store's watchdog
 * flips it to error later if no echo arrives, so the two never race.
 *
 * `launchConversation` stamps the LLM profile before rethrowing, so only the
 * plugin snapshot remains here — the launch's own snapshot runs after
 * `createConversation` resolves, which is past the point the send fails.
 */
function recoverConversationFirstMessage(
  error: ConversationFirstMessageError,
  queryClient: QueryClient,
  snapshot: PluginSnapshotContext,
): Promise<CreateConversationResponse> {
  const text = messageContentText(error.content);
  if (text.trim()) {
    const store = useOptimisticUserMessageStore.getState();
    const pendingId = store.enqueuePendingMessage({
      conversationId: error.conversationId,
      text,
      content: text,
    });
    if (isDefinitiveSendRejection(error.cause)) {
      store.markPendingMessageError(pendingId, error.message);
    }
  }
  return snapshotPlugins(error.conversationId, queryClient, snapshot).then(
    () => ({
      conversation_id: error.conversationId,
      session_api_key: null,
      url: null,
    }),
  );
}

/** The launch context the post-create plugin snapshot needs. */
interface PluginSnapshotContext {
  /** Explicit plugins attached at creation (e.g. the /launch flow). */
  explicitPlugins: PluginSpec[];
  backendId: string;
  orgId: string | null | undefined;
}

/**
 * Snapshot the conversation's plugins into client-side metadata so the
 * in-conversation plugins view can show what's loaded (coordinates only —
 * strip parameters, which may carry secrets). The agent-server doesn't return
 * a live conversation's loaded plugins, so this snapshot is the source for
 * that view. Two sources, deduped by coordinates:
 *   1. plugins explicitly attached at creation (e.g. the /launch flow);
 *   2. enabled installed plugins, which the SDK auto-loads into every new
 *      local conversation (see use-set-plugin-enabled).
 */
async function snapshotPlugins(
  conversationId: string,
  queryClient: QueryClient,
  { explicitPlugins, backendId, orgId }: PluginSnapshotContext,
): Promise<void> {
  let attachedPlugins: PluginSpec[] = explicitPlugins;
  let installed: InstalledPluginInfo[] = [];
  try {
    installed = await queryClient.ensureQueryData({
      // Same key `usePlugins` writes. Unscoped, this served the previous
      // backend's list and attached plugins the target server lacks.
      queryKey: [...PLUGINS_QUERY_KEYS.installed, backendId, orgId],
      queryFn: () => PluginsManagementService.listInstalledPlugins(),
    });
  } catch {
    // Best-effort: never let plugin lookup block conversation creation.
  }
  const seen = new Set(explicitPlugins.map(pluginReferenceKey));
  const enabledInstalled = installed
    .filter((plugin) => plugin.enabled)
    .map((plugin) => ({
      source: plugin.source,
      ref: plugin.resolved_ref ?? null,
      repo_path: plugin.repo_path ?? null,
      // Keep the human-friendly name so the plugins view shows e.g.
      // "city-weather" rather than deriving "local" from the source.
      name: plugin.name,
    }))
    .filter((plugin) => !seen.has(pluginReferenceKey(plugin)));
  attachedPlugins = [...explicitPlugins, ...enabledInstalled];
  // Merge so the metadata the launch just persisted is preserved. Best-effort:
  // the snapshot is a convenience for the plugins view, so a storage write that
  // fails (e.g. quota) must not fail the create — on the first-message recovery
  // path it would otherwise replace the recoverable error and strand the
  // created conversation.
  if (attachedPlugins.length) {
    try {
      mergeStoredConversationMetadata(conversationId, {
        plugins: attachedPlugins,
      });
    } catch (error) {
      console.warn(
        `Could not snapshot plugins for conversation ${conversationId}:`,
        error,
      );
    }
  }
}

interface UseCreateConversationOptions {
  /**
   * Set by callers that report a failed create themselves, so the global
   * mutation error toast doesn't repeat it.
   */
  disableToast?: boolean;
}

export const useCreateConversation = ({
  disableToast = false,
}: UseCreateConversationOptions = {}) => {
  const queryClient = useQueryClient();
  const { trackConversationCreated } = useTracking();
  // Warm the profile caches — the launch awaits the same queries via
  // ensureQueryData, so a warm cache makes home-launch instant. The hooks'
  // maybe-unresolved data is deliberately NOT read at launch time: activation
  // is pointer-only (it never writes agent_settings), so racing a cold cache
  // into the agent_settings fallback would silently launch the wrong agent.
  const { backend, orgId } = useActiveBackend();
  useLlmProfiles();
  useAgentProfiles();
  // Read the "Run on first message" toggle from the warmed settings
  // query so the cloud path can stamp the router instruction without a
  // settings round-trip at conversation creation. The local path reads the
  // toggle from its own settings fetch inside the encrypted-settings builder.
  const { data: settings } = useSettings();
  // The active meta-profile gates the suffix on the cloud path: with no
  // active router the agent-server does not attach `route_task_to_model`, so
  // emitting the instruction would tell the agent to call a tool it lacks.
  const { data: metaProfiles } = useMetaProfiles();

  return useMutation({
    mutationKey: CREATE_CONVERSATION_MUTATION_KEY,
    meta: { disableToast },
    mutationFn: async (
      variables: CreateConversationVariables,
    ): Promise<CreateConversationResponse> => {
      const {
        query,
        automationSetup,
        automationSetupTags,
        conversationInstructions,
        plugins,
        repository,
        workingDir,
        workspaceMode,
        parentConversationId,
        agentType,
        agentProfileId,
        onSetupScriptProgress,
        llmProfile,
      } = variables;

      // The active AgentProfile is the default launch profile for new
      // conversations (#3727), on both local and cloud. sandboxId is unused
      // here.
      // The post-create metadata writes (plugin snapshot below, LLM-profile
      // stamp inside `launchConversation`) run after `createConversation`
      // resolves — the same point a withheld-first-message send can fail.
      // Carry what they need so the recovery path can replay the snapshot.
      const snapshot: PluginSnapshotContext = {
        explicitPlugins: plugins?.map(toPluginCoordinates) ?? [],
        backendId: backend.id,
        orgId,
      };
      let conversation: Awaited<ReturnType<typeof launchConversation>>;
      try {
        conversation = await launchConversation(
          {
            queryClient,
            backendId: backend.id,
            orgId,
            isCloud: backend.kind === "cloud",
            agentProfileId,
          },
          {
            initialUserMsg: query,
            automationSetup,
            ...(automationSetupTags ? { automationSetupTags } : {}),
            conversationInstructions,
            plugins,
            metadata: repository
              ? {
                  selected_repository: repository.name,
                  selected_branch: repository.branch ?? null,
                  git_provider: repository.gitProvider,
                }
              : null,
            workingDirOverride: workingDir,
            workspaceMode,
            parentConversationId,
            agentType,
            onSetupScriptProgress,
            ...(llmProfile ? { llmProfile } : {}),
            // Cloud only; a local launch reads the toggle itself and gates it
            // on the launched profile's router. Only stamped when on, so the
            // e2e snapshot tests that assert the exact createConversation
            // payload stay stable. The active meta-profile gates the suffix:
            // no router attached means no route-at-start instruction.
            ...(settings?.run_router_at_conversation_start
              ? {
                  runRouterAtConversationStart: true,
                  hasActiveMetaProfile: !!metaProfiles?.active_meta_profile,
                }
              : {}),
          },
        );
      } catch (error) {
        // The workspace bootstrap splits create from the first send, so the
        // conversation can exist with its first turn never started. Recover on
        // that conversation instead of reporting a failed create the caller
        // would retry into a duplicate.
        if (error instanceof ConversationFirstMessageError) {
          return recoverConversationFirstMessage(error, queryClient, snapshot);
        }
        throw error;
      }

      // Cloud conversations have no app_conversation_id until the sandbox is
      // READY.
      const localConversationId = conversation.app_conversation_id;
      // Snapshot the conversation's plugins into client-side metadata so the
      // in-conversation plugins view can show what's loaded. The agent-server
      // doesn't return a live conversation's loaded plugins, so this snapshot
      // is the source for that view.
      if (localConversationId) {
        await snapshotPlugins(localConversationId, queryClient, snapshot);
      }

      // OpenHands cloud pattern: when the start task isn't immediately
      // READY (cloud sandbox is still provisioning),
      // app_conversation_id is null. We return a `task-{id}` URL so the
      // conversation route's useTaskPolling can drive it to READY and
      // then redirect to the real `/conversations/{app_conversation_id}`.
      const conversationId = conversation.app_conversation_id
        ? conversation.app_conversation_id
        : `task-${conversation.id}`;

      return {
        conversation_id: conversationId,
        session_api_key: null,
        url: conversation.agent_server_url,
        task_id: conversation.id,
      };
    },
    onSuccess: async (data, variables) => {
      trackConversationCreated({
        conversationId: data.conversation_id,
        taskId: data.task_id,
        hasRepository: !!variables.repository,
        gitProvider: variables.repository?.gitProvider,
        hasWorkspace: !!variables.workingDir,
        workspaceMode: variables.workspaceMode,
        hasInitialQuery: !!variables.query,
        agentType: variables.agentType,
        hasParentConversation: !!variables.parentConversationId,
        entryPoint: variables.entryPoint,
      });

      // Invalidate (rather than remove) so the existing paginated list stays
      // rendered while a background refetch picks up the new conversation.
      // `removeQueries` would wipe the cache and force the panel back to its
      // initial loading state, dropping loaded pages and scroll position.
      queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      // The cloud path returns a start task (no app_conversation_id
      // yet); the sidebar surfaces those via `useStartTasks` which doesn't
      // poll, so invalidate it explicitly so the in-flight task shows up
      // in the conversation list immediately.
      queryClient.invalidateQueries({
        queryKey: ["start-tasks"],
      });
    },
  });
};
