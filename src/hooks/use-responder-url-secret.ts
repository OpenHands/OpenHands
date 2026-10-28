import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { SecretsService } from "#/api/secrets-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { openhandsT } from "#/i18n/openhands-translate";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

const OPENHANDS_URL_SECRET_NAME = "OPENHANDS_URL";

/**
 * Provides an operation that ensures the active backend has an OPENHANDS_URL
 * secret without overwriting an existing value.
 */
export function useResponderUrlSecret() {
  const activeBackend = useActiveBackend();
  const queryClient = useQueryClient();
  const { i18n } = useTranslation("openhands");
  const { mutateAsync: createSecret } = useCreateSecret();

  return useCallback(async (): Promise<boolean> => {
    let secrets;
    try {
      secrets = await queryClient.fetchQuery({
        queryKey: ["secrets", activeBackend.backend.id, activeBackend.orgId],
        queryFn: SecretsService.getSecretsOrThrow,
        staleTime: 0,
        retry: false,
        meta: { disableToast: true },
      });
    } catch (error) {
      const message =
        retrieveAxiosErrorMessage(error) ||
        openhandsT(i18n, I18nKey.ERROR$GENERIC);
      if (i18n && typeof i18n.t === "function") {
        displayErrorToast(message, i18n);
      } else {
        displayErrorToast(message);
      }
      return false;
    }

    if (secrets.some((secret) => secret.name === OPENHANDS_URL_SECRET_NAME)) {
      return true;
    }

    try {
      await createSecret({
        name: OPENHANDS_URL_SECRET_NAME,
        value: window.location.origin,
      });
    } catch {
      // The shared mutation cache displays the error toast.
      return false;
    }

    await queryClient.invalidateQueries({ queryKey: ["secrets"] });
    return true;
  }, [
    activeBackend.backend.id,
    activeBackend.orgId,
    createSecret,
    i18n,
    queryClient,
  ]);
}
