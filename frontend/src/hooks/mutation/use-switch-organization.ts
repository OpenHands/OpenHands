import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useMatch, useNavigate } from "react-router";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { I18nKey } from "#/i18n/declaration";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import { setSelectedOrg } from "#/utils/local-storage";

export const useSwitchOrganization = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { setOrganizationId } = useSelectedOrganizationId();
  const navigate = useNavigate();
  const conversationMatch = useMatch("/conversations/:conversationId");

  return useMutation({
    mutationFn: ({
      orgId,
    }: {
      orgId: string;
      orgName: string;
      isPersonal: boolean;
    }) => organizationService.switchOrganization({ orgId }),
    onSuccess: (_, { orgId, orgName, isPersonal }) => {
      const message = isPersonal
        ? t(I18nKey.ORG$SWITCHED_TO_PERSONAL_WORKSPACE)
        : t(I18nKey.ORG$SWITCHED_TO_ORGANIZATION, { name: orgName });
      displaySuccessToast(message);
      // Invalidate the target org's /me query to ensure fresh data on every switch
      queryClient.invalidateQueries({
        queryKey: ["organizations", orgId, "me"],
      });
      // Update local state - this triggers automatic refetch for all org-scoped queries
      // since their query keys include organizationId (e.g., ["settings", orgId], ["secrets", orgId])
      setOrganizationId(orgId);
      // Broadcast org change to other apps (e.g. Automations) via localStorage
      setSelectedOrg(orgId);
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
