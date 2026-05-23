import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { openHands } from "#/api/open-hands-axios";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

interface ConfigureIntegrationData {
  workspace: string;
  // May be empty for Jira DC auto-enroll mode; the server generates one.
  webhookSecret: string;
  serviceAccountEmail: string;
  serviceAccountApiKey: string;
  // Jira DC only: one-time admin PAT to auto-install the webhook. Never stored.
  adminApiKey?: string;
  isActive: boolean;
}

export function useConfigureIntegration(
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
    mutationFn: async (data: ConfigureIntegrationData) => {
      const input: Record<string, unknown> = {
        workspace_name: data.workspace,
        svc_acc_email: data.serviceAccountEmail,
        is_active: data.isActive,
      };
      // Omit an empty service-account PAT so the server keeps the stored one
      // when editing (Jira DC); required server-side for a new workspace.
      if (data.serviceAccountApiKey) {
        input.svc_acc_api_key = data.serviceAccountApiKey;
      }
      // Omit an empty webhook secret so the server generates one (Jira DC
      // auto-enroll); send it verbatim otherwise.
      if (data.webhookSecret) {
        input.webhook_secret = data.webhookSecret;
      }
      // Only present for Jira DC auto-enroll; used once server-side, never stored.
      if (data.adminApiKey) {
        input.admin_api_key = data.adminApiKey;
      }

      const response = await openHands.post(
        `/integration/${platform}/workspaces`,
        input,
      );

      const { success, redirect, authorizationUrl } = response.data;

      if (success) {
        if (redirect) {
          if (authorizationUrl) {
            window.location.href = authorizationUrl;
          } else {
            throw new Error("Could not get authorization URL from the server.");
          }
        } else {
          window.location.reload();
        }
      } else {
        throw new Error("Configuration failed");
      }

      return response.data;
    },
    onSuccess: () => {
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
