import { useQuery } from "@tanstack/react-query";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type { BitbucketDCResourcesResponse } from "#/api/integration-service/integration-service.types";

export function useBitbucketDCResources(enabled: boolean = true) {
  return useQuery<BitbucketDCResourcesResponse>({
    queryKey: ["bitbucket-dc-resources"],
    queryFn: () => integrationService.getBitbucketDCResources(),
    enabled,
    staleTime: 1000 * 60 * 2,
    gcTime: 1000 * 60 * 10,
  });
}
