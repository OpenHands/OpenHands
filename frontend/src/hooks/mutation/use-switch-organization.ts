import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useMatch, useNavigate } from "react-router";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useSwitchOrganization = () => {
  const queryClient = useQueryClient();
  const { setOrganizationId } = useSelectedOrganizationId();
  const navigate = useNavigate();
  const conversationMatch = useMatch("/conversations/:conversationId");

  return useMutation({
    mutationFn: (orgId: string) =>
      organizationService.switchOrganization({ orgId }),
    onSuccess: (_, orgId) => {
      // Invalidate the target org's /me query to ensure fresh data on every switch
      queryClient.invalidateQueries({
        queryKey: ["organizations", orgId, "me"],
      });
      // Update local state
      setOrganizationId(orgId);
      // Invalidate settings for the new org context
      queryClient.invalidateQueries({ queryKey: ["settings"] });
      // Invalidate conversations to fetch data for the new org context
      queryClient.invalidateQueries({ queryKey: ["user", "conversations"] });
      // Remove all individual conversation queries to clear any stale/null data
      // from the previous org context
      queryClient.removeQueries({ queryKey: ["user", "conversation"] });

      // Redirect to home if on a conversation page since org context has changed
      if (conversationMatch) {
        navigate("/");
      }
    },
  });
};
