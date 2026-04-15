import { useQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useMe = () => {
  const { data: config } = useConfig();
  const { organizationId } = useSelectedOrganizationId();

  const isSaas = config?.app_mode === "saas";

  return useQuery({
    queryKey: ["organizations", organizationId, "me"],
    queryFn: () => organizationService.getMe({ orgId: organizationId! }),
    staleTime: 1000 * 60 * 5, // 5 minutes
    enabled: isSaas && !!organizationId,
  });
};
