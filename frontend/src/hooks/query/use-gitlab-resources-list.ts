import { useQuery } from "@tanstack/react-query";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type { GitLabResourcesResponse } from "#/api/integration-service/integration-service.types";

/**
 * Hook to fetch GitLab resources with webhook status
 */
export function useGitLabResources(enabled: boolean = true) {
  return useQuery<GitLabResourcesResponse>({
    queryKey: ["gitlab-resources"],
    queryFn: () => integrationService.getGitLabResources(),
    enabled,
    staleTime: 1000 * 60 * 2, // 2 minutes
    gcTime: 1000 * 60 * 10, // 10 minutes
  });
}
