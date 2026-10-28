import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_QUERY_KEYS,
  AGENT_PROFILES_RETRY_OPTIONS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { invalidateConversationQueries } from "./conversation-mutation-utils";
import { useModelStore } from "#/stores/model-store";

interface SwitchAcpModelVars {
  /**
   * When set, the ACP conversation's running model is swapped live via the
   * wrapper's ``session/set_model`` (POST /switch_acp_model) and the user's
   * saved default is untouched. When null (home page / no session), the model
   * is persisted as the default the next conversation launches with.
   */
  conversationId: string | null;
  /** Null saves no model, so the next conversation starts on the agent's own default. */
  model: string | null;
}

/**
 * ACP analog of {@link useSwitchLlmProfile}. Switches the ACP model
 * per-conversation when called from inside a conversation (live in-place model
 * switch; a created-but-not-yet-run conversation is a persist-only deferral on
 * the server, so blank conversations work too); persists it as the launch
 * default when called from the home page.
 *
 * The home-page persist target is the **active ACP AgentProfile**: profile
 * launches resolve the agent server-side purely from the stored profile
 * (``agent_profile_id`` and ``agent_settings`` are mutually exclusive launch
 * sources), so writing ``agent_settings.acp_model`` would not affect them.
 * Only when the profile list resolves with no active ACP profile does the
 * write fall back to the ``agent_settings_diff`` a Cloud launch without a
 * profile reads. A failed profile discovery propagates instead of downgrading
 * (#16523): an active-profile launch ignores agent_settings, so persisting
 * the pick there would silently drop it.
 *
 * Invalidates the same conversation/settings query keys the profile hook does
 * so the chat-input model chip + conversation chip refresh with the new model.
 */
export const useSwitchAcpModel = () => {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();

  return useMutation({
    // A home-page ACP model pick (no conversation) is an explicit model
    // choice made before the first message: mark it synchronously here (not in
    // mutationFn/onSuccess) so a quick send reads the flag before the async
    // profile listing resolves — otherwise the launch races the pick, routes
    // anyway, and the stale flag set by the late onSuccess leaks to the next
    // conversation.
    onMutate: ({ conversationId }) => {
      if (!conversationId) {
        useModelStore.getState().setHomeManualModelPick(true);
      }
      return null;
    },
    mutationFn: async ({ conversationId, model }: SwitchAcpModelVars) => {
      if (conversationId) {
        if (model) {
          await AgentServerConversationService.switchAcpModel(
            conversationId,
            model,
          );
        }
        return;
      }

      // Home page: resolve the active profile from the shared query cache
      // (same keys/retry policy as useCreateConversation's launch path, so a
      // pick can't disagree with what a subsequent send would launch). Like
      // that launch path, do not fall back to the legacy agent_settings
      // persist when profile discovery fails (#16523): an active-profile
      // launch ignores agent_settings, so the pick would be silently dropped.
      // The rejection surfaces via the global mutation error toast.
      const agentProfiles: AgentProfileListResponse =
        await queryClient.ensureQueryData({
          queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backend.id, orgId],
          queryFn: AgentProfilesService.listProfiles,
          ...AGENT_PROFILES_RETRY_OPTIONS,
        });
      const activeProfile = agentProfiles.profiles?.find(
        (profile) =>
          profile.id != null &&
          profile.id === agentProfiles.active_agent_profile_id,
      );

      if (activeProfile?.agent_kind === "acp") {
        // Merge over the stored profile so a model-only pick can't wipe the
        // fields this surface doesn't model (command, session mode, …).
        const detail = await queryClient.ensureQueryData({
          queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
            backend.id,
            orgId,
            activeProfile.name,
          ),
          queryFn: () => AgentProfilesService.getProfile(activeProfile.name),
          ...AGENT_PROFILES_RETRY_OPTIONS,
        });
        await AgentProfilesService.saveProfile(
          activeProfile.name,
          mergeAgentProfileSaveInput(detail.profile, {
            agent_kind: "acp",
            acp_model: model,
          }),
        );
        return;
      }

      // No active ACP profile: persist as the agent-settings default a Cloud
      // launch without a profile reads. The backend deep-merges
      // ``agent_settings_diff`` into the existing ``agent_settings`` dict, so
      // a scalar ``acp_model`` diff updates only the model and preserves the
      // selected provider + command.
      await SettingsService.saveSettings({
        agent_settings_diff: { acp_model: model },
      });
    },
    onSuccess: (_data, { conversationId }) => {
      if (conversationId) {
        invalidateConversationQueries(queryClient, conversationId);
      } else {
        // Mirror useSwitchLlmProfile's home-page path: clear the stale settings
        // cache so the next conversation-start reads the newly saved default,
        // and refetch the settings query so the home-page chip updates. The
        // agent-profiles prefix covers both the list and the detail entry the
        // profile-persist path just rewrote.
        SettingsService.invalidateCache();
        queryClient.invalidateQueries({
          queryKey: SETTINGS_QUERY_KEYS.personal(),
        });
        queryClient.invalidateQueries({
          queryKey: AGENT_PROFILES_QUERY_KEYS.all,
        });
      }
    },
    // No meta.disableToast: unlike useSwitchLlmProfile (which tailors a
    // "Switched to {name} failed" message via its own onError), there's no
    // per-action message worth tailoring here, so the global mutation error
    // toast reports a failed switch / profile or settings write rather than
    // swallowing it.
    onError: (_error, { conversationId }) => {
      // A failed home pick never persisted: clear the flag so the next launch
      // routes as normal instead of inheriting a stale suppress.
      if (!conversationId) {
        useModelStore.getState().setHomeManualModelPick(false);
      }
    },
  });
};
