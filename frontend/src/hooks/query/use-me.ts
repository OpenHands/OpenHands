import { useQuery } from "@tanstack/react-query";
import { useConfig } from "./use-config";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useMe = () => {
  const { data: config } = useConfig();
  const { organizationId } = useSelectedOrganizationId();

  const isSaas = config?.APP_MODE === "saas";

  return useQuery({
    queryKey: ["organizations", organizationId, "me"],
    queryFn: () => organizationService.getMe({ orgId: organizationId! }),
    enabled: isSaas && !!organizationId,
  });
};
