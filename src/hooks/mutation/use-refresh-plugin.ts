import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import PluginsManagementService from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Update an installed plugin from its source. Version / resolved coordinates may
 * change, so the installed list is invalidated on success.
 */
export function useRefreshPlugin() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    mutationFn: (name: string) => PluginsManagementService.refreshPlugin(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEYS.installed });
      displaySuccessToast(t(I18nKey.SETTINGS$PLUGINS_REFRESH_SUCCESS));
    },
    onError: (error) => {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
      );
    },
  });
}
