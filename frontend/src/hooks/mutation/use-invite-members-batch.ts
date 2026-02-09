import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useInviteMembersBatch = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: ({ emails }: { emails: string[] }) => {
      if (!organizationId) {
        throw new Error("Organization ID is required");
      }
      return organizationService.inviteMembers({
        orgId: organizationId,
        emails,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", organizationId],
      });
    },
  });
};
