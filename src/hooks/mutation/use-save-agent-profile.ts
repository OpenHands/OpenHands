import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService, {
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { AGENT_PROFILES_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useSaveAgentProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      name,
      profile,
    }: {
      name: string;
      profile: AgentProfileSaveInput;
    }) => AgentProfilesService.saveProfile(name, profile),
    onSuccess: async () => {
      // Prefix match invalidates every backend/org-suffixed list key.
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch + targeted toasts (409/422).
    meta: { disableToast: true },
  });
}
