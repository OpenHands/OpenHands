import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type {
  ResourceIdentifier,
  ResourceInstallationResult,
} from "#/api/integration-service/integration-service.types";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/**
 * Hook to reinstall webhook on a specific resource
 */
export function useReinstallGitLabWebhook() {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation<
    ResourceInstallationResult,
    Error,
    ResourceIdentifier,
    unknown
  >({
    mutationFn: (resource: ResourceIdentifier) =>
      integrationService.reinstallGitLabWebhook({ resource }),
    onSuccess: (data) => {
      // Invalidate and refetch the resources list
      queryClient.invalidateQueries({ queryKey: ["gitlab-resources"] });

      if (data.success) {
        displaySuccessToast(t(I18nKey.GITLAB$WEBHOOK_REINSTALL_SUCCESS));
      } else if (data.error) {
        displayErrorToast(data.error);
      } else {
        displayErrorToast(t(I18nKey.GITLAB$WEBHOOK_REINSTALL_FAILED));
      }
    },
    onError: (error) => {
      const errorMessage =
        error?.message || t(I18nKey.GITLAB$WEBHOOK_REINSTALL_FAILED);
      displayErrorToast(errorMessage);
    },
  });
}
