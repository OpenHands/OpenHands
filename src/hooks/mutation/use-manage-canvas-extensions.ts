import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstallCanvasExtensionRequest } from "#/types/canvas-extension";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorBody, getApiErrorMessage } from "#/utils/api-error-message";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

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
    // The default toast prints the client's raw `HTTP 400: {...}` message;
    // the server's `detail` says which of Source/Ref/Path is actually wrong.
    meta: { disableToast: true },
    mutationFn: (request: InstallCanvasExtensionRequest) =>
      CanvasExtensionsService.install(request),
    onSuccess: () => {
      void invalidate();
      displaySuccessToast(
        t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_INSTALL_SUCCESS),
      );
    },
    onError: (error) => {
      // A transport failure carries no response body, so keep the shared
      // "Disconnected" wording for it and use the server's detail otherwise.
      const message = getApiErrorBody(error)
        ? getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC))
        : retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC);
      displayErrorToast(message);
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
