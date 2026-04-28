import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useDeleteOrganization = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { organizationId, setOrganizationId } = useSelectedOrganizationId();

  return useMutation({
    mutationFn: () => {
      if (!organizationId) throw new Error("Organization ID is required");
      return organizationService.deleteOrganization({ orgId: organizationId });
    },
    onSuccess: () => {
      // Remove stale cache BEFORE clearing the selected organization.
      // This prevents useAutoSelectOrganization from using the old currentOrgId
      // when it runs during the re-render triggered by setOrganizationId(null).
      // Using removeQueries (not invalidateQueries) ensures stale data is gone immediately.
      queryClient.removeQueries({
        queryKey: ["organizations"],
        exact: true,
      });
      queryClient.removeQueries({
        queryKey: ["organizations", organizationId],
      });

      // Now clear the selected organization - useAutoSelectOrganization will
      // wait for fresh data since the cache is empty
      setOrganizationId(null);

      navigate("/");
    },
  });
};
