import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import { AGENT_PROFILES_QUERY_KEYS } from "#/hooks/query/query-keys";

interface RenameAgentProfileVariables {
  name: string;
  newName: string;
}

export function useRenameAgentProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, newName }: RenameAgentProfileVariables) =>
      AgentProfilesService.renameProfile(name, newName),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
    },
    // Consumers handle errors with try-catch and manual toasts.
    meta: { disableToast: true },
  });
}
