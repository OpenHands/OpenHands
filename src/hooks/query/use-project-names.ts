import { useQuery } from "@tanstack/react-query";

import ProjectsService from "#/api/projects-service/projects-service.api";

/** Project ids compare without dashes or case: tags store the plain hex form. */
export const projectIdKey = (id: string) => id.replace(/-/g, "").toLowerCase();

/**
 * Names of the active backend's projects, keyed by {@link projectIdKey}.
 * Empty on a backend without Projects, so callers fall back to the raw id.
 */
export function useProjectNames(): Map<string, string> {
  const { data } = useQuery({
    queryKey: ["projects", "names"],
    queryFn: async () => {
      try {
        return await ProjectsService.listProjects();
      } catch {
        return [];
      }
    },
    meta: { disableToast: true },
    staleTime: 60_000,
  });
  return new Map((data ?? []).map((p) => [projectIdKey(p.id), p.name]));
}
