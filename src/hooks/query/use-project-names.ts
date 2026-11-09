import { useQuery } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";

import ProjectsService from "#/api/projects-service/projects-service.api";

/** Project ids compare without dashes or case: tags store the plain hex form. */
export const projectIdKey = (id: string) => id.replace(/-/g, "").toLowerCase();

/**
 * Names of the active backend's projects, keyed by {@link projectIdKey}, and
 * whether the backend has Projects at all. A backend with Projects turned off
 * (or without them) answers 404, so `enabled` is false and callers hide
 * project tags; on other errors the names are empty and callers fall back to
 * the raw id.
 */
export function useProjectNames(): {
  names: Map<string, string>;
  enabled: boolean;
} {
  const { data } = useQuery({
    queryKey: ["projects", "names"],
    queryFn: async () => {
      try {
        return {
          enabled: true,
          projects: await ProjectsService.listProjects(),
        };
      } catch (error) {
        return {
          enabled: !(error instanceof HttpError && error.status === 404),
          projects: [],
        };
      }
    },
    meta: { disableToast: true },
    staleTime: 60_000,
  });
  return {
    names: new Map(
      (data?.projects ?? []).map((p) => [projectIdKey(p.id), p.name]),
    ),
    enabled: data?.enabled ?? true,
  };
}
