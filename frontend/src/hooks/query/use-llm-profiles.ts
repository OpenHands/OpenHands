import { useQuery } from "@tanstack/react-query";
import ProfilesService from "#/api/settings-service/profiles-service.api";
import { useIsAuthed } from "./use-is-authed";

export const LLM_PROFILES_QUERY_KEY = "llm-profiles";

export function useLlmProfiles() {
  const { data: userIsAuthenticated } = useIsAuthed();

  return useQuery({
    queryKey: [LLM_PROFILES_QUERY_KEY],
    queryFn: ProfilesService.listProfiles,
    enabled: !!userIsAuthenticated,
    retry: (_, error) => error.status !== 404,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    meta: { disableToast: true },
  });
}
