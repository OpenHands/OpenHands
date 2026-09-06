import { useQuery } from "@tanstack/react-query";
import SkillsService from "#/api/skills-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SkillInfo } from "#/types/settings";

/**
 * @param projectDir Workspace root to load project skills from. Conversation
 *   views pass the conversation's own workspace so the catalog matches the
 *   skills loaded into that conversation; the global Skills page omits it.
 */
export const useSkills = (projectDir?: string) => {
  // getSkills resolves against the active backend, so the cache entry has to
  // name it. setActive does not blanket-invalidate on a switch; it requires
  // long-lived queries to carry the backend id and orgId instead.
  const active = useActiveBackend();
  return useQuery<SkillInfo[]>({
    queryKey: ["skills", projectDir ?? null, active.backend.id, active.orgId],
    queryFn: () => SkillsService.getSkills(projectDir),
    staleTime: 1000 * 60 * 10, // 10 minutes – skill list rarely changes
    refetchOnWindowFocus: false,
  });
};
