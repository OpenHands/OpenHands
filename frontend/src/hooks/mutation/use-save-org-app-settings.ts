import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  organizationService,
  OrganizationAppSettingsUpdate,
} from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { ORGANIZATION_APP_SETTINGS_KEYS } from "#/hooks/query/query-keys";

export const useSaveOrgAppSettings = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async (settings: OrganizationAppSettingsUpdate) =>
      organizationService.saveOrganizationAppSettings(settings),
    onSuccess: () => {
      // Invalidate org app settings cache for the current org
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
