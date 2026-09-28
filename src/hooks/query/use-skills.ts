import { useQuery } from "@tanstack/react-query";
import SkillsService from "#/api/skills-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SkillInfo } from "#/types/settings";
import { SKILLS_QUERY_KEYS } from "./query-keys";

/**
 * @param projectDir Workspace root to load project skills from. Conversation
 *   views pass the conversation's own workspace so the catalog matches the
 *   skills loaded into that conversation; the global Skills page omits it.
 */
export const useSkills = (projectDir?: string) => {
  const { backend, orgId } = useActiveBackend();

  return useQuery<SkillInfo[]>({
    queryKey: SKILLS_QUERY_KEYS.catalog(backend.id, orgId, projectDir ?? null),
    queryFn: () => SkillsService.getSkills(projectDir),
    staleTime: 1000 * 60 * 10, // 10 minutes – skill list rarely changes
    refetchOnWindowFocus: false,
  });
};
