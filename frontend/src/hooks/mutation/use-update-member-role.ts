import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { OrganizationUserRole } from "#/types/org";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useUpdateMemberRole = () => {
  const queryClient = useQueryClient();
  const { orgId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async ({
      userId,
      role,
    }: {
      userId: string;
      role: OrganizationUserRole;
    }) => {
      if (!orgId) {
        throw new Error("Organization ID is required to update member role");
      }
      return organizationService.updateMemberRole({
        orgId,
        userId,
        role,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations", "members"] });
    },
  });
};
