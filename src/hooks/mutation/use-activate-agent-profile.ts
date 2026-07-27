import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  AGENT_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";

/**
 * Shared key so any picker instance can observe an in-flight activation via
 * `useIsMutating` (the pill button and the menu that fires it are separate
 * hook instances, so per-observer `isPending` wouldn't line up).
 */
export const ACTIVATE_AGENT_PROFILE_MUTATION_KEY = ["activate-agent-profile"];

/**
 * Activate an agent profile by its stable UUID `id`. Activation is
 * pointer-only (it does NOT write agent_settings), but it changes the launch
 * default the backend resolves for new conversations, so the settings cache is
 * invalidated defensively alongside the profiles list.
 *
 * The active pointer is flipped optimistically so the picker label and the
 * launch default (`useCreateConversation` reads `active_agent_profile_id`)
 * reflect the selection before the refetch lands. Errors roll the pointer back
 * and surface via the global mutation toast (no `meta.disableToast`) so a
 * failure isn't silent when fired from the chat-input picker, whose menu
 * unmounts on select.
 */
export function useActivateAgentProfile() {
  const queryClient = useQueryClient();

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
    onSuccess: async () => {
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
