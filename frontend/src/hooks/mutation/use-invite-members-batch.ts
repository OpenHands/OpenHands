import { useMutation, useQueryClient } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useInviteMembersBatch = () => {
  const queryClient = useQueryClient();
  const { orgId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: ({ emails }: { emails: string[] }) =>
      organizationService.inviteMembers({ orgId: orgId!, emails }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", orgId],
      });
    },
  });
};