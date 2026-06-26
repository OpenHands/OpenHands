import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_APP_SETTINGS_KEY } from "#/hooks/query/query-keys";

export const useOrganizationAppSettings = () =>
  useQuery({
    queryKey: ORGANIZATION_APP_SETTINGS_KEY,
    queryFn: () => organizationService.getOrganizationAppSettings(),
  });
