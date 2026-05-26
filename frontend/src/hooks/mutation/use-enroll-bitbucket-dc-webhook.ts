import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type {
  BitbucketDCResourceIdentifier,
  BitbucketDCWebhookEnrollmentResult,
} from "#/api/integration-service/integration-service.types";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

export function useEnrollBitbucketDCWebhook() {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation<
    BitbucketDCWebhookEnrollmentResult,
    Error,
    BitbucketDCResourceIdentifier,
    unknown
  >({
    mutationFn: (resource: BitbucketDCResourceIdentifier) =>
      integrationService.enrollBitbucketDCWebhook({ resource }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["bitbucket-dc-resources"] });

      if (data.success) {
        displaySuccessToast(
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLL_SUCCESS),
        );
      } else if (data.error) {
        displayErrorToast(data.error);
      } else {
        displayErrorToast(
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLL_FAILED),
        );
      }
    },
    onError: (error) => {
      displayErrorToast(
        error?.message ||
          t(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLL_FAILED),
      );
    },
  });
}
