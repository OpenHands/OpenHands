import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { SuggestedTask } from "#/utils/types";
import { AgentKind, Provider } from "#/types/settings";
import { useTracking } from "#/hooks/use-tracking";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import PluginsManagementService, {
  type InstalledPluginInfo,
} from "#/api/plugins-management-service";
import {
  PLUGINS_QUERY_KEYS,
  LLM_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
} from "#/hooks/query/query-keys";
import { pluginReferenceKey } from "#/utils/plugin-display";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
  toPluginCoordinates,
  type WorkspaceMode,
} from "#/api/conversation-metadata-store";

export interface CreateConversationVariables {
  query?: string;
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
  entryPoint?: string; // analytics only; not forwarded to the service
}

export const CREATE_CONVERSATION_MUTATION_KEY = ["create-conversation"];

interface CreateConversationResponse {
  conversation_id: string;
  session_api_key: string | null;
  url: string | null;
  task_id?: string;
}

export const useCreateConversation = () => {
  const queryClient = useQueryClient();
  const { trackConversationCreated } = useTracking();
  // Cache-warm on the home page (the profile picker reads the same query).
  // Stamped onto the conversation at creation so the switcher can show the
  // exact profile even when several profiles share a model (#1082).
  const { data: llmProfiles } = useLlmProfiles();
  // Warm the agent-profiles cache too — the launch path below awaits the same
  // query via ensureQueryData, so a warm cache makes home-launch instant. The
  // hook's maybe-unresolved data is deliberately NOT read at launch time:
  // activation is pointer-only (it never writes agent_settings), so racing a
  // cold cache into the agent_settings fallback would silently launch the
  // wrong agent.
  const { backend, orgId } = useActiveBackend();
  useAgentProfiles();

  return useMutation({
    mutationKey: CREATE_CONVERSATION_MUTATION_KEY,
    mutationFn: async (
      variables: CreateConversationVariables,
    ): Promise<CreateConversationResponse> => {
      const {
        query,
        conversationInstructions,
        plugins,
        repository,
        workingDir,
        workspaceMode,
        parentConversationId,
        agentType,
        agentProfileId,
      } = variables;

      // The active AgentProfile is the default launch profile for new
      // conversations (#3727), on both local and cloud (cloud gained
      // /api/agent-profiles in OpenHands #15060, #3730). Await the list from
      // the shared query cache: a send fired before the home query resolves
      // must still launch from the active profile, not fall through to the
      // agent_settings path. Degrades safely: if the fetch errors (older
      // backend without the surface), this stays undefined and creation falls
      // back to the encrypted agent_settings launch path.
      let agentProfiles: AgentProfileListResponse | undefined;
      try {
        agentProfiles = await queryClient.ensureQueryData({
          queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backend.id, orgId],
          queryFn: AgentProfilesService.listProfiles,
          ...AGENT_PROFILES_RETRY_OPTIONS,
        });
      } catch {
        // Profiles unavailable → legacy agent_settings launch.
      }

      const requestedAgentProfileId =
        agentProfileId ?? agentProfiles?.active_agent_profile_id ?? undefined;

      // Fall back to the legacy agent_settings launch when the resolved agent
      // profile can't resolve its LLM. The agent-server seeds a `default`
      // openhands profile whose `llm_profile_ref` can point at an LLM profile
      // that doesn't exist (fresh store, or one configured with named profiles
      // only); launching from it 404s ("LLM profile '<ref>' not found") and
      // would brick home-launch. agent_settings reflects the active LLM, so the
      // fallback degrades cleanly until the seed mirrors it (SDK #3933).
      // ACP profiles carry no llm_profile_ref, so they're never gated here.
      const resolvedAgentProfile = requestedAgentProfileId
        ? agentProfiles?.profiles?.find(
            (profile) => profile.id === requestedAgentProfileId,
          )
        : undefined;
      // Cloud has no `agent_settings` payload to fall back to — the downgrade
      // below only makes sense on local, where it exists and carries the
      // canvas-only enrichments. Gating it here keeps cloud always launching
      // from the resolved profile id, so the conversation gets
      // `launched_agent_profile` stamped and the profile's config applied
      // (#1571 review).
      const isCloud = backend.kind === "cloud";
      let effectiveAgentProfileId = requestedAgentProfileId;
      if (
        !isCloud &&
        resolvedAgentProfile?.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME &&
        resolvedAgentProfile?.agent_kind === "openhands"
      ) {
        // The seeded OpenHands `default` profile is the enriched baseline, not a
        // deliberate profile pick — it mirrors global agent_settings. Launch it
        // via agent_settings so the canvas-only enrichments the profile-resolution
        // path drops survive for the common home-launch: the <RUNTIME_SERVICES>
        // system-message suffix and project-skill loading (buildAgentContext).
        // Named profiles are deliberate custom configs and still use the profile
        // path (accepting that enrichment boundary).
        // Trade-off: per-profile fields set on `default` itself don't apply on
        // home-launch — custom per-profile config belongs in a named profile.
        //
        // Scoped to OpenHands: an ACP `default` must keep the profile path.
        // Activation is pointer-only, so global agent_settings is stale (often
        // still OpenHands) when an ACP profile is active — launching it via
        // agent_settings would start the wrong agent. ACP carries no
        // <RUNTIME_SERVICES> enrichment, so there's nothing to preserve.
        //
        // Scoped to local: cloud never writes agent_settings, so it always
        // resolves `default` server-side via agent_profile_id (validated below).
        effectiveAgentProfileId = undefined;
      } else if (
        resolvedAgentProfile?.agent_kind === "openhands" &&
        resolvedAgentProfile.llm_profile_ref
      ) {
        // Await the LLM-profile list rather than reading the maybe-unresolved
        // `useLlmProfiles()` result: a send fired before that query loads (or
        // after it errors) must still validate the ref, not launch blind.
        let llmProfileExists = false;
        try {
          const llm = await queryClient.ensureQueryData({
            queryKey: [...LLM_PROFILES_QUERY_KEYS.all, backend.id, orgId],
            queryFn: ProfilesService.listProfiles,
            // Match the agent-profiles fetch above: on a backend where this
            // errors, fall back to agent_settings immediately rather than
            // stalling the send through the default exponential backoff.
            retry: false,
          });
          llmProfileExists = llm.profiles.some(
            (profile) => profile.name === resolvedAgentProfile.llm_profile_ref,
          );
        } catch {
          // List unavailable → can't validate → fall back to agent_settings.
        }
        if (!llmProfileExists) {
          // Downgrade is silent in the UI; leave a diagnosable trace.
          console.warn(
            `Agent profile "${resolvedAgentProfile.name}" references missing ` +
              `LLM profile "${resolvedAgentProfile.llm_profile_ref}"; ` +
              "launching from agent_settings instead.",
          );
          effectiveAgentProfileId = undefined;
        }
      }

      // Only extend the call with the profile tail when launching from a
      // profile, so a plain create stays byte-identical to the legacy
      // agent_settings path (#3727). sandboxId is unused here.
      // TODO(#1587): createConversation has grown to 11 positional params;
      // refactor it to an options object so this position-skipping tail isn't
      // needed.
      const profileArgs: [undefined, string, AgentKind | undefined] | [] =
        effectiveAgentProfileId
          ? [
              undefined,
              effectiveAgentProfileId,
              resolvedAgentProfile?.agent_kind,
            ]
          : [];

      const conversation =
        await AgentServerConversationService.createConversation(
          query,
          conversationInstructions,
          plugins,
          repository
            ? {
                selected_repository: repository.name,
                selected_branch: repository.branch ?? null,
                git_provider: repository.gitProvider,
              }
            : null,
          workingDir,
          workspaceMode,
          parentConversationId,
          agentType,
          ...profileArgs,
        );

      // Stamp the active LLM profile onto the (local) conversation so the
      // chat switcher shows the exact profile even when several profiles
      // share a model (#1082). Cloud conversations don't use local profiles
      // (app_conversation_id stays null until the sandbox is READY). Merge so
      // the repo/workspace metadata the service just persisted is preserved.
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
            queryKey: PLUGINS_QUERY_KEYS.installed,
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
      // A launch from a named OpenHands profile runs that profile's
      // `llm_profile_ref`, which can differ from the standalone active LLM
      // profile — stamp the ref so the switcher pill names the exact profile
      // the conversation runs (#1082). The agent_settings path (the `default`
      // baseline or a dangling ref, where effectiveAgentProfileId is cleared)
      // runs the active LLM, so it keeps `active_profile`. ACP profiles carry
      // no LLM profile, so they fall through to the active-profile stamp
      // (unused by the ACP model chip).
      const activeProfile =
        effectiveAgentProfileId &&
        resolvedAgentProfile?.agent_kind === "openhands"
          ? resolvedAgentProfile.llm_profile_ref
          : (llmProfiles?.active_profile ?? null);
      if (localConversationId && (activeProfile || attachedPlugins.length)) {
        const prev = getStoredConversationMetadata(localConversationId);
        setStoredConversationMetadata(localConversationId, {
          selected_repository: prev?.selected_repository ?? null,
          selected_branch: prev?.selected_branch ?? null,
          git_provider: prev?.git_provider ?? null,
          selected_workspace: prev?.selected_workspace ?? null,
          workspace_mode: prev?.workspace_mode ?? null,
          active_profile: activeProfile ?? prev?.active_profile ?? null,
          plugins: attachedPlugins.length
            ? attachedPlugins
            : (prev?.plugins ?? null),
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
