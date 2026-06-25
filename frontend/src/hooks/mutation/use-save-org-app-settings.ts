import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  organizationService,
  OrganizationAppSettingsUpdate,
} from "#/api/organization-service/organization-service.api";
import { ORGANIZATION_SETTINGS_KEY } from "#/hooks/query/query-keys";

export const useSaveOrgAppSettings = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      orgId,
      settings,
    }: {
      orgId: string;
      settings: OrganizationAppSettingsUpdate;
    }) => organizationService.saveOrganizationAppSettings({ orgId, settings }),
    onSuccess: () => {
      // Invalidate org app settings cache for the specific org
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_SETTINGS_KEY,
      });
    },
    meta: {
      disableToast: true,
    },
  });
};
