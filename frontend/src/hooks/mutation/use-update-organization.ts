import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useUpdateOrganization = () => {
  const queryClient = useQueryClient();
  const { orgId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: (name: string) => {
      if (!orgId) throw new Error("Organization ID is required");
      return organizationService.updateOrganization({ orgId, name });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations", orgId] });
    },
  });
};
