import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useRemoveMember = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: ({ userId }: { userId: string }) => {
      if (!organizationId) {
        throw new Error("Organization ID is required");
      }
      return organizationService.removeMember({
        orgId: organizationId,
        userId,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", organizationId],
      });
    },
  });
};
