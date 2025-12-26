import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useUpdateOrganization = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: (name: string) => {
      if (!organizationId) throw new Error("Organization ID is required");
      return organizationService.updateOrganization({
        orgId: organizationId,
        name,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["organizations", organizationId],
      });
    },
  });
};
