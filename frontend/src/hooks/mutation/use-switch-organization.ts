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
      // Update local state
      setOrganizationId(orgId);
      // Refetch getMe for the new organization
      queryClient.invalidateQueries({
        queryKey: ["organizations", orgId, "me"],
      });
    },
  });
};
