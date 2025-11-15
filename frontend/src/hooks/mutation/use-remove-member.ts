import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useRemoveMember = () => {
  const queryClient = useQueryClient();
  const { orgId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: ({ userId }: { userId: string }) =>
      organizationService.removeMember({ orgId: orgId!, userId }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", orgId],
      });
    },
  });
};