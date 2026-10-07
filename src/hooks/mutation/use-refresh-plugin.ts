import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import PluginsManagementService from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayApiErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/**
 * Update an installed plugin from its source. Version / resolved coordinates may
 * change, so the installed list is invalidated on success.
 */
export function useRefreshPlugin() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    // This hook toasts the server's reason itself; skip the global toast.
    meta: { disableToast: true },
    mutationFn: (name: string) => PluginsManagementService.refreshPlugin(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEYS.installed });
      displaySuccessToast(t(I18nKey.SETTINGS$PLUGINS_REFRESH_SUCCESS));
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}
