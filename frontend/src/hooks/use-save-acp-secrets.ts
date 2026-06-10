import React from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { I18nKey } from "#/i18n/declaration";
import { useCreateSecret } from "#/hooks/mutation/use-create-secret";
import type { ACPProviderSecretField } from "#/constants/acp-provider-secrets";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Persists each filled ACP credential field as a global secret and refreshes
 * the secret query cache. ``saveFilled`` resolves ``true`` on success (or when
 * nothing needed saving) and ``false`` on error so callers can gate navigation.
 * ``silent`` suppresses the success toast when the caller emits its own.
 */
export function useSaveAcpSecrets(fields: ACPProviderSecretField[]) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { mutateAsync: createSecret } = useCreateSecret();
  const [isSaving, setIsSaving] = React.useState(false);

  const saveFilled = async (
    values: Record<string, string>,
    { silent = false }: { silent?: boolean } = {},
  ) => {
    const toSave = fields
      .map((field) => ({ name: field.name, value: values[field.name]?.trim() }))
      .filter((entry): entry is { name: string; value: string } =>
        Boolean(entry.value),
      );
    if (toSave.length === 0) return true;

    setIsSaving(true);
    try {
      await Promise.all(
        toSave.map(({ name, value }) => createSecret({ name, value })),
      );
      await queryClient.invalidateQueries({ queryKey: ["secrets-search"] });
      await queryClient.invalidateQueries({ queryKey: ["secrets"] });
      if (!silent) displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      return true;
    } catch (error) {
      const message = retrieveAxiosErrorMessage(error as AxiosError);
      displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
      return false;
    } finally {
      setIsSaving(false);
    }
  };

  return { saveFilled, isSaving };
}
