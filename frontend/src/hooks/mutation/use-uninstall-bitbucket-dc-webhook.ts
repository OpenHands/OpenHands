import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type {
  BitbucketDCResourceIdentifier,
  BitbucketDCWebhookInstallationResult,
} from "#/api/integration-service/integration-service.types";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

export function useUninstallBitbucketDCWebhook() {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation<
    BitbucketDCWebhookInstallationResult,
    Error,
    BitbucketDCResourceIdentifier,
    unknown
  >({
    mutationFn: (resource: BitbucketDCResourceIdentifier) =>
      integrationService.uninstallBitbucketDCWebhook({ resource }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["bitbucket-dc-resources"] });

      if (data.success) {
        displaySuccessToast(
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_UNINSTALL_SUCCESS),
        );
      } else if (data.error) {
        displayErrorToast(data.error);
      } else {
        displayErrorToast(
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_UNINSTALL_FAILED),
        );
      }
    },
    onError: (error) => {
      displayErrorToast(
        error?.message ||
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_UNINSTALL_FAILED),
      );
    },
  });
}
