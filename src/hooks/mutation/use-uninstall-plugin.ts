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
 * Uninstall a plugin. The plugin returns to "available" in the catalog, so both
 * the installed list and the marketplace catalog are invalidated on success.
 */
export function useUninstallPlugin() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    mutationFn: (name: string) =>
      PluginsManagementService.uninstallPlugin(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEYS.installed });
      queryClient.invalidateQueries({
        queryKey: PLUGINS_QUERY_KEYS.marketplace,
      });
      displaySuccessToast(t(I18nKey.SETTINGS$PLUGINS_UNINSTALL_SUCCESS));
    },
    onError: (error) => {
      displayErrorToast(
        retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
      );
    },
  });
}
