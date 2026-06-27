import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { openHands } from "#/api/open-hands-axios";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

interface UpdateJiraDcWorkspaceStatusData {
  workspace: string;
  isActive: boolean;
}

export function useUpdateJiraDcWorkspaceStatus({
  onSettled,
}: {
  onSettled: () => void;
}) {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMutation({
    mutationFn: async (data: UpdateJiraDcWorkspaceStatusData) => {
      const response = await openHands.post(
        "/integration/jira-dc/workspaces/status",
        {
          workspace_name: data.workspace,
          is_active: data.isActive,
        },
      );

      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["integration-status", "jira-dc"],
      });
    },
    onError: (error) => {
      const errorMessage = retrieveAxiosErrorMessage(error);
      displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
    },
    onSettled,
  });
}
