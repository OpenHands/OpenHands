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
import { agentProfileDetailQueryKey } from "#/hooks/query/use-active-acp-profile-detail";
import { invalidateConversationQueries } from "./conversation-mutation-utils";

interface SwitchAcpModelVars {
  /**
   * When set, the ACP conversation's running model is swapped live via the
   * wrapper's ``session/set_model`` (POST /switch_acp_model) and the user's
   * saved default is untouched. When null (home page / no session), the model
   * is persisted as the default the next conversation launches with.
   */
  conversationId: string | null;
  model: string;
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
 * Only when no ACP profile is active (a legacy backend without the
 * /api/agent-profiles surface) does the write fall back to the
 * ``agent_settings_diff`` the legacy launch path reads.
 *
 * Invalidates the same conversation/settings query keys the profile hook does
 * so the chat-input model chip + conversation chip refresh with the new model.
 */
export const useSwitchAcpModel = () => {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();

  return useMutation({
    mutationFn: async ({ conversationId, model }: SwitchAcpModelVars) => {
      if (conversationId) {
        await AgentServerConversationService.switchAcpModel(
          conversationId,
          model,
        );
        return;
      }

      // Home page: resolve the active profile from the shared query cache
      // (same keys/retry policy as useCreateConversation's launch path, so a
      // pick can't disagree with what a subsequent send would launch).
      let agentProfiles: AgentProfileListResponse | undefined;
      try {
        agentProfiles = await queryClient.ensureQueryData({
          queryKey: [...AGENT_PROFILES_QUERY_KEYS.all, backend.id, orgId],
          queryFn: AgentProfilesService.listProfiles,
          ...AGENT_PROFILES_RETRY_OPTIONS,
        });
      } catch {
        // Profiles unavailable → legacy agent_settings persist below.
      }
      const activeProfile = agentProfiles?.profiles?.find(
        (profile) =>
          profile.id != null &&
          profile.id === agentProfiles?.active_agent_profile_id,
      );

      if (activeProfile?.agent_kind === "acp") {
        // Merge over the stored profile so a model-only pick can't wipe the
        // fields this surface doesn't model (command, session mode, …).
        const detail = await queryClient.ensureQueryData({
          queryKey: agentProfileDetailQueryKey(
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

      // Legacy/no-profile backends: persist as the agent-settings default. The
      // backend deep-merges ``agent_settings_diff`` into the existing
      // ``agent_settings`` dict, so a scalar ``acp_model`` diff updates only
      // the model and preserves the selected provider + command.
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
  });
};
