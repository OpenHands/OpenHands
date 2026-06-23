import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  organizationService,
  OrganizationAppSettingsUpdate,
} from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_APP_SETTINGS_KEY } from "#/hooks/query/query-keys";

export const useSaveOrgAppSettings = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: OrganizationAppSettingsUpdate) =>
      organizationService.saveOrganizationAppSettings(settings),
    onSuccess: () => {
      // Invalidate org app settings cache
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_APP_SETTINGS_KEY,
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
