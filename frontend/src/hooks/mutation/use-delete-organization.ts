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
      queryClient.invalidateQueries({ queryKey: ["organizations"] });
      setOrganizationId(null);
      navigate("/");
    },
  });
};
