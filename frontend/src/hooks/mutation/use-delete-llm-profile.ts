import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import { LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-llm-profiles";

export function useDeleteLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (name: string) => {
      await ProfilesService.deleteProfile(name);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LLM_PROFILES_QUERY_KEY] });
      // Deleting the active profile clears ``llm_profiles.active`` server-side;
      // the settings cache must refetch or the LLM page will keep showing
      // the deleted profile as in-use.
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
}
