import { useQuery } from "@tanstack/react-query";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type { AzureDevOpsResourcesResponse } from "#/api/integration-service/integration-service.types";

export function useAzureDevOpsResources(enabled: boolean = true) {
  return useQuery<AzureDevOpsResourcesResponse>({
    queryKey: ["azure-devops-resources"],
    queryFn: () => integrationService.getAzureDevOpsResources(),
    enabled,
    staleTime: 1000 * 60 * 2,
    gcTime: 1000 * 60 * 10,
  });
}
