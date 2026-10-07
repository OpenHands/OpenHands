import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import PluginsManagementService, {
  type InstallPluginRequest,
} from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayApiErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/**
 * Install a plugin from a git source or local path. Installing flips a catalog
 * entry from available to installed, so both the installed list and the
 * marketplace catalog are invalidated on success.
 */
export function useInstallPlugin() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    // This hook toasts the server's reason itself; skip the global toast.
    meta: { disableToast: true },
    mutationFn: (request: InstallPluginRequest) =>
      PluginsManagementService.installPlugin(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEYS.installed });
      queryClient.invalidateQueries({
        queryKey: PLUGINS_QUERY_KEYS.marketplace,
      });
      displaySuccessToast(t(I18nKey.SETTINGS$PLUGINS_INSTALL_SUCCESS));
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}
