import { useQuery } from "@tanstack/react-query";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_APP_SETTINGS_KEYS } from "#/hooks/query/query-keys";

export const useOrganizationAppSettings = () => {
  const { organizationId } = useSelectedOrganizationId();

  return useQuery({
    queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
    queryFn: () => organizationService.getOrganizationAppSettings(),
    enabled: !!organizationId,
  });
};
