import { useQuery } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";

import ProjectsService from "#/api/projects-service/projects-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useIsExperimentalFeatureEnabled } from "./use-experimental-features";

/** The experimental feature that turns Projects on, per backend. */
export const PROJECTS_FEATURE_KEY = "projects";

/**
 * Every Projects query sits under ["projects", backend, revision], so one
 * invalidation of {@link projectsQueryRoot} refreshes them all, and switching
 * backends never shows the previous backend's projects.
 */
export const projectsQueryRoot = (backend: {
  id: string;
  connectionRevision?: number;
}) => ["projects", backend.id, backend.connectionRevision ?? 0] as const;

const isNotFound = (error: unknown) =>
  error instanceof HttpError && error.status === 404;

/**
 * Whether the active backend has Projects turned on (Settings › Experimental).
 * Everything project-related in Canvas hides while this is false.
 */
export function useProjectsEnabled(): boolean {
  const { backend } = useActiveBackend();
  const enabled = useIsExperimentalFeatureEnabled(PROJECTS_FEATURE_KEY);
  return backend.kind === "local" && enabled;
}

/** The active backend's projects, oldest first. */
export function useProjects() {
  const { backend } = useActiveBackend();
  const enabled = useProjectsEnabled();
  return useQuery({
    queryKey: [...projectsQueryRoot(backend), "list"],
    queryFn: () => ProjectsService.listProjects(),
    enabled,
    staleTime: 15_000,
    meta: { disableToast: true },
    retry: (count, error) => !isNotFound(error) && count < 2,
  });
}

export function useProject(projectId: string | undefined) {
  const { backend } = useActiveBackend();
  const enabled = useProjectsEnabled();
  return useQuery({
    queryKey: [...projectsQueryRoot(backend), "project", projectId],
    queryFn: () => ProjectsService.getProject(projectId!),
    enabled: enabled && !!projectId,
    meta: { disableToast: true },
    retry: (count, error) => !isNotFound(error) && count < 2,
  });
}

/**
 * The project's board: its coordinator and its other conversations, grouped
 * by what they need from the user. Polled, since workers change state on
 * their own.
 */
export function useProjectConversations(projectId: string | undefined) {
  const { backend } = useActiveBackend();
  const enabled = useProjectsEnabled();
  return useQuery({
    queryKey: [...projectsQueryRoot(backend), "conversations", projectId],
    queryFn: () => ProjectsService.listConversations(projectId!),
    enabled: enabled && !!projectId,
    refetchInterval: 5_000,
    refetchIntervalInBackground: false,
    meta: { disableToast: true },
  });
}

export function useProjectArtifacts(projectId: string | undefined) {
  const { backend } = useActiveBackend();
  const enabled = useProjectsEnabled();
  return useQuery({
    queryKey: [...projectsQueryRoot(backend), "artifacts", projectId],
    queryFn: () => ProjectsService.listArtifacts(projectId!),
    enabled: enabled && !!projectId,
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,
    meta: { disableToast: true },
  });
}

export function useProjectArtifactText(
  projectId: string | undefined,
  name: string | null,
  updatedAt: string | undefined,
) {
  const { backend } = useActiveBackend();
  return useQuery({
    queryKey: [
      ...projectsQueryRoot(backend),
      "artifact-text",
      projectId,
      name,
      updatedAt,
    ],
    queryFn: () => ProjectsService.readArtifactText(projectId!, name!),
    enabled: !!projectId && !!name,
    staleTime: Infinity,
    meta: { disableToast: true },
  });
}
