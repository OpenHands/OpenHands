import { projectIdKey } from "#/api/projects-service/projects-service.api";
import { useProjects, useProjectsEnabled } from "./use-projects";

export { projectIdKey };

/**
 * Names of the active backend's projects, keyed by {@link projectIdKey}, and
 * whether the backend has Projects turned on. While it is off, callers hide
 * project tags; while the names load or fail, they fall back to the raw id.
 */
export function useProjectNames(): {
  names: Map<string, string>;
  enabled: boolean;
} {
  const enabled = useProjectsEnabled();
  const { data } = useProjects();
  return {
    names: new Map((data ?? []).map((p) => [projectIdKey(p.id), p.name])),
    enabled,
  };
}
