import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export const useRemoveMember = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: ({ userId }: { userId: string }) => {
      if (!organizationId) {
        throw new Error("Organization ID is required");
      }
      return organizationService.removeMember({
        orgId: organizationId,
        userId,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.ORG$REMOVE_MEMBER_SUCCESS));
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", organizationId],
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ORG$REMOVE_MEMBER_ERROR));
    },
  });
};
