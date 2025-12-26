import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { OrganizationUserRole } from "#/types/org";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useUpdateMemberRole = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: async ({
      userId,
      role,
    }: {
      userId: string;
      role: OrganizationUserRole;
    }) => {
      if (!organizationId) {
        throw new Error("Organization ID is required to update member role");
      }
      return organizationService.updateMember({
        orgId: organizationId,
        userId,
        role,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["organizations", "members"] });
    },
  });
};
