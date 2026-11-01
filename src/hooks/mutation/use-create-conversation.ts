import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
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
import { pluginReferenceKey } from "#/utils/plugin-display";
import { useModelStore } from "#/stores/model-store";
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
  // Cloud only: start inside this existing sandbox instead of provisioning
  // one (the sidebar's "New Conversation in Sandbox"). Local backends have no
  // sandboxes, and the local launch path ignores it.
  sandboxId?: string;
  // Launch from a specific AgentProfile (local backend). When omitted, the
  // active AgentProfile (if any) is used so home-composed conversations
  // launch from the user's selected profile (#3727).
  agentProfileId?: string;
  // Cloud: run on this LLM profile (a member's home pick) without activating it.
  llmProfile?: string;
  entryPoint?: string; // analytics only; not forwarded to the service
}

export const CREATE_CONVERSATION_MUTATION_KEY = ["create-conversation"];

interface CreateConversationResponse {
  conversation_id: string;
  session_api_key: string | null;
  url: string | null;
  task_id?: string;
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
        llmProfile,
        sandboxId,
      } = variables;

      // A home-page model pick (LLM profile or ACP model) made before the
      // first message is an explicit choice the router shouldn't override on
      // turn 1 — suppress the suffix for this single launch without touching
      // the global setting. The flag is set at the UI pick handlers (the
      // single chokepoint) and read here. It is cleared in `onSuccess` (not
      // here) so a rejected launch doesn't consume it — a retry after a
      // failed launch still suppresses, and the flag can't leak to an
      // unrelated launch because a successful launch always clears it.
      const suppressRouterAtStart =
        useModelStore.getState().homeManualModelPick;

      // The active AgentProfile is the default launch profile for new
      // conversations (#3727), on both local and cloud.
      const conversation = await launchConversation(
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
          ...(llmProfile ? { llmProfile } : {}),
          ...(sandboxId ? { sandboxId } : {}),
          // Cloud only; a local launch reads the toggle itself and gates it
          // on the launched profile's router. Only stamped when on, so the
          // e2e snapshot tests that assert the exact createConversation
          // payload stay stable. The active meta-profile gates the suffix:
          // no router attached means no route-at-start instruction.
          ...(settings?.run_router_at_conversation_start &&
          !suppressRouterAtStart
            ? {
                runRouterAtConversationStart: true,
                hasActiveMetaProfile: !!metaProfiles?.active_meta_profile,
              }
            : {}),
          // Local: the builder gates the route-at-start suffix on this rather
          // than reading the store, so child/`/new` launches (which don't pass
          // it) are unaffected by a pending home pick. Only threaded when a
          // pick was made so the payload-snapshot tests stay byte-identical
          // otherwise.
          ...(suppressRouterAtStart ? { suppressRouterAtStart } : {}),
        },
      );

      // Cloud conversations have no app_conversation_id until the sandbox is
      // READY.
      const localConversationId = conversation.app_conversation_id;
      // Snapshot the conversation's plugins into client-side metadata so the
      // in-conversation plugins view can show what's loaded (coordinates only
      // — strip parameters, which may carry secrets). The agent-server doesn't
      // return a live conversation's loaded plugins, so this snapshot is the
      // source for that view. Two sources, deduped by coordinates:
      //   1. plugins explicitly attached at creation (e.g. the /launch flow);
      //   2. enabled installed plugins, which the SDK auto-loads into every new
      //      local conversation (see use-set-plugin-enabled).
      const explicitPlugins = plugins?.map(toPluginCoordinates) ?? [];
      let attachedPlugins: PluginSpec[] = explicitPlugins;
      if (localConversationId) {
        let installed: InstalledPluginInfo[] = [];
        try {
          installed = await queryClient.ensureQueryData({
            // Same key `usePlugins` writes. Unscoped, this served the previous
            // backend's list and attached plugins the target server lacks.
            queryKey: [...PLUGINS_QUERY_KEYS.installed, backend.id, orgId],
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
      }
      // Merge so the metadata the launch just persisted is preserved.
      if (localConversationId && attachedPlugins.length) {
        mergeStoredConversationMetadata(localConversationId, {
          plugins: attachedPlugins,
        });
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
      // The home-manual-pick flag was read (not cleared) in the mutation fn
      // so a rejected launch wouldn't consume it. Now that the launch
      // succeeded, clear it so it can't leak to a later, unrelated launch.
      useModelStore.getState().setHomeManualModelPick(false);
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
