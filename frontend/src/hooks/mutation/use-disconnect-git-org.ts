import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { I18nKey } from "#/i18n/declaration";
import { gitClaimsQueryKey } from "#/hooks/query/use-git-organizations";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export const useDisconnectGitOrg = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: ({ claimId }: { claimId: string }) => {
      if (!organizationId) throw new Error("Organization ID is required");
      return organizationService.disconnectGitOrg({
        orgId: organizationId,
        claimId,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.ORG$DISCONNECT_SUCCESS));
      queryClient.invalidateQueries({
        queryKey: gitClaimsQueryKey(organizationId),
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ORG$DISCONNECT_ERROR));
    },
    meta: { disableToast: true },
  });
};
