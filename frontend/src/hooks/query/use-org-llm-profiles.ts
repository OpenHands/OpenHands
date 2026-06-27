import { useQuery } from "@tanstack/react-query";
import OrgProfilesService from "#/api/organization-service/org-profiles-service.api";
import { useIsAuthed } from "./use-is-authed";

export const ORG_LLM_PROFILES_QUERY_KEY = "org-llm-profiles";

export function useOrgLlmProfiles(orgId: string | null | undefined) {
  const { data: userIsAuthenticated } = useIsAuthed();

  return useQuery({
    queryKey: [ORG_LLM_PROFILES_QUERY_KEY, orgId],
    queryFn: () => OrgProfilesService.listProfiles(orgId!),
    enabled: !!userIsAuthenticated && !!orgId,
    retry: (_, error) => error.status !== 404,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    meta: { disableToast: true },
  });
}
