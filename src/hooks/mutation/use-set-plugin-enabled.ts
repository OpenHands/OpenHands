import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import PluginsManagementService from "#/api/plugins-management-service";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Enable or disable an installed plugin. Enabled installed plugins auto-load
 * into new conversations (via the SDK auto-load wiring), so the enabled flag is
 * the enforced source of truth — invalidate the installed list on success.
 */
export function useSetPluginEnabled() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    // This hook toasts the server's reason itself; skip the global toast.
    meta: { disableToast: true },
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      PluginsManagementService.setPluginEnabled(name, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEYS.installed });
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}
