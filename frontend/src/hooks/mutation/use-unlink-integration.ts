import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { openHands } from "#/api/open-hands-axios";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export function useUnlinkIntegration(
  platform: "jira" | "jira-dc" | "linear",
  {
    onSettled,
  }: {
    onSettled: () => void;
  },
) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    // adminApiKey is Jira DC only: the integration owner tearing it
    // down may pass a one-time PAT so the Jira webhook is revoked too. Never
    // stored. Omitted for the non-owner self-disconnect path.
    mutationFn: (adminApiKey?: string) =>
      openHands.post(
        `/integration/${platform}/workspaces/unlink`,
        adminApiKey ? { admin_api_key: adminApiKey } : {},
      ),
    onSuccess: (response, adminApiKey) => {
      const webhookRemoveFailed =
        platform === "jira-dc" &&
        !!adminApiKey?.trim() &&
        response.data?.webhookRemoved === false;

      if (webhookRemoveFailed) {
        displayErrorToast(
          t(I18nKey.PROJECT_MANAGEMENT$JIRA_DC_WEBHOOK_REMOVE_FAILED),
        );
      } else {
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      }
      queryClient.invalidateQueries({
        queryKey: ["integration-status", platform],
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
    },
    onSettled,
  });
}
