import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useOrgConversationStats = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ["organizations", "conversation-stats", organizationId],
    queryFn: () =>
      organizationService.getConversationStats({ orgId: organizationId! }),
    enabled: !!organizationId,
  });
};
