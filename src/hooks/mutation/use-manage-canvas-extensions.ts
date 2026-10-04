import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type {
  InstallCanvasExtensionRequest,
  InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { clearPinnedExtensionRoute } from "#/hooks/use-pinned-home-route";
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

// A transport failure carries no response body, so keep the shared
// "Disconnected" wording for it and use the server's detail otherwise.
function displayInstallError(error: unknown, fallback: string) {
  displayErrorToast(
    getApiErrorBody(error)
      ? getApiErrorMessage(error, fallback)
      : retrieveAxiosErrorMessage(error) || fallback,
  );
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
      displaySuccessToast(t(I18nKey.SETTINGS$APPS_INSTALL_SUCCESS));
    },
    onError: (error) => displayInstallError(error, t(I18nKey.ERROR$GENERIC)),
  });
}

/**
 * Re-install an App from its recorded source. The server validates the new
 * copy before replacing the old one and carries over its enabled state.
 */
export function useRefreshCanvasExtension() {
  const invalidate = useInvalidateCanvasExtensions();
  const { t } = useTranslation("openhands");
  return useMutation({
    meta: { disableToast: true },
    mutationFn: (extension: InstalledCanvasExtensionInfo) =>
      CanvasExtensionsService.install({
        source: extension.source,
        ref: extension.requested_ref,
        repo_path: extension.repo_path,
        force: true,
      }),
    onSuccess: () => {
      void invalidate();
      displaySuccessToast(t(I18nKey.SETTINGS$APPS_REFRESH_SUCCESS));
    },
    onError: (error) => displayInstallError(error, t(I18nKey.ERROR$GENERIC)),
  });
}

export function useSetCanvasExtensionEnabled() {
  const invalidate = useInvalidateCanvasExtensions();
  return useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      CanvasExtensionsService.setEnabled(name, enabled),
    onSuccess: (_data, { name, enabled }) => {
      if (!enabled) {
        clearPinnedExtensionRoute(name);
      }
      void invalidate();
    },
  });
}

export function useUninstallCanvasExtension() {
  const invalidate = useInvalidateCanvasExtensions();
  const { t } = useTranslation("openhands");
  return useMutation({
    mutationFn: (name: string) => CanvasExtensionsService.uninstall(name),
    onSuccess: (_data, name) => {
      clearPinnedExtensionRoute(name);
      void invalidate();
      displaySuccessToast(t(I18nKey.SETTINGS$APPS_UNINSTALL_SUCCESS));
    },
  });
}
