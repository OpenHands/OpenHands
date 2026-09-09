import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import ProjectsService from "#/api/projects-service/projects-service.api";
import type {
  CreateProjectPayload,
  CreateWorktreePayload,
} from "#/api/projects-service/projects-types";
import { PROJECTS_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useProjects() {
  return useQuery({
    queryKey: PROJECTS_QUERY_KEYS.list(),
    queryFn: () => ProjectsService.listProjects(),
  });
}

export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: PROJECTS_QUERY_KEYS.detail(projectId ?? ""),
    queryFn: () => ProjectsService.getProject(projectId!),
    enabled: Boolean(projectId),
  });
}

function useInvalidateProjects(projectId?: string | null) {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEYS.all });
    if (projectId) {
      queryClient.invalidateQueries({
        queryKey: PROJECTS_QUERY_KEYS.detail(projectId),
      });
    }
  };
}

export function useCreateProject() {
  const invalidate = useInvalidateProjects();
  return useMutation({
    mutationFn: (payload: CreateProjectPayload) =>
      ProjectsService.createProject(payload),
    onSuccess: invalidate,
  });
}

export function useCreateWorktree(projectId: string) {
  const invalidate = useInvalidateProjects(projectId);
  return useMutation({
    mutationFn: (payload: CreateWorktreePayload) =>
      ProjectsService.createWorktree(projectId, payload),
    onSuccess: invalidate,
  });
}

export function useRemoveWorktree(projectId: string) {
  const invalidate = useInvalidateProjects(projectId);
  return useMutation({
    mutationFn: (worktreeId: string) =>
      ProjectsService.removeWorktree(projectId, worktreeId),
    onSuccess: invalidate,
  });
}

export function useAssignWorktree(projectId: string) {
  const invalidate = useInvalidateProjects(projectId);
  return useMutation({
    mutationFn: ({
      worktreeId,
      agentSessionId,
    }: {
      worktreeId: string;
      agentSessionId: string;
    }) => ProjectsService.assignWorktree(projectId, worktreeId, agentSessionId),
    onSuccess: invalidate,
  });
}
