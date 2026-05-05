import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import { LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-llm-profiles";

interface RenameLlmProfileVariables {
  name: string;
  newName: string;
}

export function useRenameLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ name, newName }: RenameLlmProfileVariables) => {
      await ProfilesService.renameProfile(name, newName);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LLM_PROFILES_QUERY_KEY] });
      // Renaming the active profile changes ``llm_profiles.active`` to the
      // new name; the settings cache must refetch so any UI that reads the
      // active-profile name stays in sync.
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
}
