import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { OrganizationUserRole } from "#/types/org";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export const useUpdateMemberRole = () => {
  const queryClient = useQueryClient();
  const { organizationId } = useSelectedOrganizationId();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: async ({
      userId,
      role,
    }: {
      userId: string;
      role: OrganizationUserRole;
    }) => {
      if (!organizationId) {
        throw new Error("Organization ID is required to update member role");
      }
      return organizationService.updateMember({
        orgId: organizationId,
        userId,
        role,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.ORG$UPDATE_ROLE_SUCCESS));
      queryClient.invalidateQueries({
        queryKey: ["organizations", "members", organizationId],
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ORG$UPDATE_ROLE_ERROR));
    },
  });
};
