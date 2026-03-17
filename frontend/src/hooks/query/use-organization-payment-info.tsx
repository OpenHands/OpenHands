import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrganizationPaymentInfo = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", organizationId, "payment"],
    queryFn: () =>
      organizationService.getOrganizationPaymentInfo({
        orgId: organizationId!,
      }),
    enabled: !!organizationId,
  });
};
