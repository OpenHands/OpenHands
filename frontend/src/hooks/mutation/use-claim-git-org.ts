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

export const useClaimGitOrg = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: ({
      provider,
      gitOrganization,
    }: {
      provider: string;
      gitOrganization: string;
    }) => {
      if (!organizationId) throw new Error("Organization ID is required");
      return organizationService.claimGitOrg({
        orgId: organizationId,
        provider,
        gitOrganization,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.ORG$CLAIM_SUCCESS));
      queryClient.invalidateQueries({
        queryKey: gitClaimsQueryKey(organizationId),
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ORG$CLAIM_ERROR));
    },
    meta: { disableToast: true },
  });
};
