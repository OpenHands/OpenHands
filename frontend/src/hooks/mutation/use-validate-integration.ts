/* eslint-disable @typescript-eslint/no-explicit-any */
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import { useTranslation } from "react-i18next";

import { openHands } from "#/api/open-hands-axios";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export function useValidateIntegration(
  platform: "jira" | "jira-dc" | "linear",
  {
    onSuccess,
    onError,
  }: {
    onSuccess: (data: any) => void;
    onError: (error: any) => void;
  },
) {
  const { t } = useTranslation();

  return useMutation({
    // The 404 from .../workspaces/validate/{name} is an expected signal that the
    // workspace isn't configured yet (handled below by advancing the form), not
    // an error. Opt out of the global MutationCache error toast so it doesn't
    // surface a spurious "Request failed with status 404"; onError still toasts
    // genuine (non-404) failures itself.
    meta: { disableToast: true },
    mutationFn: (workspace?: string) => {
      const workspaceParam = workspace ? `/${workspace}` : "";
      return openHands.get(
        `/integration/${platform}/workspaces/validate${workspaceParam}`,
      );
    },
    onSuccess,
    onError: (error) => {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        onError(error);
      } else {
        const errorMessage = retrieveAxiosErrorMessage(error);
        displayErrorToast(
          errorMessage ||
            t(I18nKey.PROJECT_MANAGEMENT$VALIDATE_INTEGRATION_ERROR),
        );
      }
    },
  });
}
