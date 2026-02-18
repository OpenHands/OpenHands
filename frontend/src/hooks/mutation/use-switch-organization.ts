import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useSwitchOrganization = () => {
  const queryClient = useQueryClient();
  const { setOrganizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: (orgId: string) =>
      organizationService.switchOrganization({ orgId }),
    onSuccess: (_, orgId) => {
      // Invalidate the target org's /me query to ensure fresh data on every switch
      queryClient.invalidateQueries({
        queryKey: ["organizations", orgId, "me"],
      });
      // Update local state
      setOrganizationId(orgId);
      // Invalidate settings for the new org context
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });
};
