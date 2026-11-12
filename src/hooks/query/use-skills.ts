import { useQuery } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import SkillsService from "#/api/skills-service";
import { SkillInfo } from "#/types/settings";

/**
 * @param projectDir Workspace root to load project skills from. Conversation
 *   views pass the conversation's own workspace so the catalog matches the
 *   skills loaded into that conversation; the global Skills page omits it.
 *
 * The query key includes the active backend's id and orgId per the
 * `ActiveBackendProvider.setActive` contract: `getSkills` resolves against
 * the active backend, so a switch must start a new query instead of serving
 * the previous backend's cached list for the stale window (#16893).
 */
export const useSkills = (
  projectDir?: string,
  options: { enabled?: boolean } = {},
) => {
  const { backend, orgId } = useActiveBackend();
  return useQuery<SkillInfo[]>({
    queryKey: ["skills", backend.id, orgId, projectDir ?? null],
    queryFn: () => SkillsService.getSkills(projectDir),
    enabled: options.enabled ?? true,
    staleTime: 1000 * 60 * 10, // 10 minutes – skill list rarely changes
    refetchOnWindowFocus: false,
  });
};
