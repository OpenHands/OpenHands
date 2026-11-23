import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { AGENT_PROFILES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { syncAgentSettingsKindAfterProfileSave } from "#/hooks/mutation/sync-active-agent-kind";

export function useSaveAgentProfile() {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();

  return useMutation({
    mutationFn: ({
      name,
      profile,
    }: {
      name: string;
      profile: AgentProfileSaveInput;
    }) => AgentProfilesService.saveProfile(name, profile),
    onSuccess: async (_response, { name }) => {
      // On cloud, a save of the currently active profile re-states the global
      // `agent_settings.agent_kind` from the saved profile (#18883), so an
      // agent-kind edit can't leave profile-less launches on the previous
      // kind. Local saves keep their no-settings-write behavior.
      await syncAgentSettingsKindAfterProfileSave(
        queryClient,
        name,
        backend.kind,
      );
      // Prefix match invalidates every backend/org-suffixed list key.
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch + targeted toasts (409/422).
    meta: { disableToast: true },
  });
}
