import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_SETTINGS_KEY } from "#/hooks/query/query-keys";

export const useOrganizationAppSettings = (orgId: string | null) =>
  useQuery({
    queryKey: [...ORGANIZATION_SETTINGS_KEY, orgId],
    queryFn: () =>
      organizationService.getOrganizationAppSettings({ orgId: orgId! }),
    enabled: !!orgId,
  });
