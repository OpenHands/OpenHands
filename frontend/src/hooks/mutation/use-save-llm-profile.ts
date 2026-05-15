import { useMutation, useQueryClient } from "@tanstack/react-query";
import ProfilesService, {
  SaveLlmProfileRequest,
} from "#/api/settings-service/profiles-service.api";
import { LLM_PROFILES_QUERY_KEY } from "#/hooks/query/use-llm-profiles";

interface SaveLlmProfileVariables {
  name: string;
  request?: SaveLlmProfileRequest;
}

export function useSaveLlmProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ name, request }: SaveLlmProfileVariables) => {
      await ProfilesService.saveProfile(name, request ?? {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [LLM_PROFILES_QUERY_KEY] });
    },
  });
}
