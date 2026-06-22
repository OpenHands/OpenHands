import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  organizationService,
  OrganizationAppSettingsUpdate,
} from "#/api/organization-service/organization-service.api";

export const useSaveOrgAppSettings = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (settings: OrganizationAppSettingsUpdate) =>
      organizationService.saveOrganizationAppSettings(settings),
    onSuccess: () => {
      // Invalidate org app settings cache
      queryClient.invalidateQueries({
        queryKey: ["organization-app-settings"],
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
