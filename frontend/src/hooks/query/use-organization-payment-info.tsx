import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrganizationPaymentInfo = () => {
  const { orgId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", orgId, "payment"],
    queryFn: () =>
      organizationService.getOrganizationPaymentInfo({ orgId: orgId! }),
    enabled: !!orgId,
  });
};
