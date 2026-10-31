import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { syncAgentSettingsKindAfterProfileActivation } from "#/hooks/mutation/sync-active-agent-kind";

/**
 * Shared key so any picker instance can observe an in-flight activation via
 * `useIsMutating` (the pill button and the menu that fires it are separate
 * hook instances, so per-observer `isPending` wouldn't line up).
 */
export const ACTIVATE_AGENT_PROFILE_MUTATION_KEY = ["activate-agent-profile"];

/**
 * Activate an agent profile by its stable UUID `id`. Activation is
 * pointer-only (it does NOT write agent_settings itself), but it changes the
 * launch default the backend resolves for new conversations, so the settings
 * cache is invalidated defensively alongside the profiles list. On cloud, a
 * successful activation also re-states the global `agent_settings.agent_kind`
 * from the activated profile (#18883) so the home switcher and profile-less
 * launches stop following a kind left behind by onboarding; local activation
 * still writes nothing.
 *
 * The active pointer is flipped optimistically so the picker label and the
 * launch default (`useCreateConversation` reads `active_agent_profile_id`)
 * reflect the selection before the refetch lands. Errors roll the pointer back
 * and surface via the global mutation toast (no `meta.disableToast`) so a
 * failure isn't silent when fired from the chat-input picker, whose menu
 * unmounts on select. A failed kind sync only toasts and never fails the
 * activation.
 */
export function useActivateAgentProfile() {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();

  return useMutation({
    mutationKey: ACTIVATE_AGENT_PROFILE_MUTATION_KEY,
    mutationFn: (profileId: string) =>
      AgentProfilesService.activateProfile(profileId),
    onMutate: async (profileId) => {
      await queryClient.cancelQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
      const snapshots = queryClient.getQueriesData<AgentProfileListResponse>({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
      queryClient.setQueriesData<AgentProfileListResponse>(
        { queryKey: AGENT_PROFILES_QUERY_KEYS.all },
        (prev) =>
          prev ? { ...prev, active_agent_profile_id: profileId } : prev,
      );
      return { snapshots };
    },
    onError: (_err, _profileId, context) => {
      context?.snapshots?.forEach(([key, data]) => {
        queryClient.setQueryData(key, data);
      });
    },
    onSuccess: async (_response, profileId) => {
      await syncAgentSettingsKindAfterProfileActivation(
        queryClient,
        profileId,
        backend.kind,
      );
      SettingsService.invalidateCache();
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.personal(),
      });
    },
  });
}
