import { useQuery } from "@tanstack/react-query";
import SkillsService from "#/api/skills-service";
import { SkillInfo } from "#/types/settings";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * @param projectDir Workspace root to load project skills from. Conversation
 *   views pass the conversation's own workspace so the catalog matches the
 *   skills loaded into that conversation; the global Skills page omits it.
 */
export const useSkills = (projectDir?: string) => {
  // The skills list is backend-scoped: Cloud backends serve it from the Cloud
  // skills API, and each local backend serves its own agent-server's user and
  // project skills. Keying by the active backend (and org) makes a backend
  // switch a brand-new query, so one backend's skills are never served from
  // another backend's cache (see ActiveBackendProvider's setActive contract).
  const active = useActiveBackend();

  return useQuery<SkillInfo[]>({
    queryKey: ["skills", active.backend.id, active.orgId, projectDir ?? null],
    queryFn: () => SkillsService.getSkills(projectDir),
    staleTime: 1000 * 60 * 10, // 10 minutes – skill list rarely changes
    refetchOnWindowFocus: false,
  });
};
