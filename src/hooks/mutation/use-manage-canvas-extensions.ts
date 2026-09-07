import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstallCanvasExtensionRequest } from "#/types/canvas-extension";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";

function useInvalidateCanvasExtensions() {
  const queryClient = useQueryClient();
  return () =>
    queryClient.invalidateQueries({
      queryKey: CANVAS_EXTENSIONS_QUERY_KEYS.all,
    });
}

export function useInstallCanvasExtension() {
  const invalidate = useInvalidateCanvasExtensions();
  const { t } = useTranslation("openhands");
  return useMutation({
    mutationFn: (request: InstallCanvasExtensionRequest) =>
      CanvasExtensionsService.install(request),
    onSuccess: () => {
      void invalidate();
      displaySuccessToast(
        t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_INSTALL_SUCCESS),
      );
    },
  });
}

export function useSetCanvasExtensionEnabled() {
  const invalidate = useInvalidateCanvasExtensions();
  return useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      CanvasExtensionsService.setEnabled(name, enabled),
    onSuccess: () => void invalidate(),
  });
}

export function useUninstallCanvasExtension() {
  const invalidate = useInvalidateCanvasExtensions();
  const { t } = useTranslation("openhands");
  return useMutation({
    mutationFn: (name: string) => CanvasExtensionsService.uninstall(name),
    onSuccess: () => {
      void invalidate();
      displaySuccessToast(
        t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_UNINSTALL_SUCCESS),
      );
    },
  });
}
