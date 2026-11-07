import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import type {
  InstallCanvasExtensionRequest,
  InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import {
  displayApiErrorToast,
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

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
      displaySuccessToast(t(I18nKey.SETTINGS$APPS_INSTALL_SUCCESS));
    },
    onError: (error) => {
      // The server's 409 says to retry with `force=true`, which the form
      // cannot do; Update or Uninstall on the existing card can.
      if (isSdkHttpStatusError(error, 409)) {
        displayErrorToast(t(I18nKey.SETTINGS$APPS_ALREADY_INSTALLED));
        return;
      }
      displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC));
    },
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
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
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
      displaySuccessToast(t(I18nKey.SETTINGS$APPS_UNINSTALL_SUCCESS));
    },
  });
}
