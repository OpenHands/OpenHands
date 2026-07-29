import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import type { MarketplaceField } from "@openhands/extensions/integrations";
import { SecretsService } from "#/api/secrets-service";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/** Truncates a list of key names for display in toasts. */
function formatKeyList(keys: string[]): string {
  const MAX = 3;
  if (keys.length <= MAX) return keys.join(", ");
  return `${keys.slice(0, MAX).join(", ")} … (+${keys.length - MAX})`;
}

/**
 * Returns a stable function that upserts checked envFields into the Secrets
 * store. Callers may ignore the returned promise for background saves or await
 * it when later work depends on the secret being present. MCP server config
 * and the Secrets store are separate — this bridges the gap so Automation
 * Server can access credentials without a separate manual step. Internally,
 * `SecretsService.createSecret` is an upsert, so existing secrets with the
 * same name are overwritten safely.
 */
export function useSaveFieldsAsSecrets() {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();

  return useCallback(
    (
      envFields: MarketplaceField[],
      values: Record<string, string>,
      savedAsSecret: Record<string, boolean>,
    ): Promise<void> => {
      const fieldsToSave = envFields.filter(
        (field) => savedAsSecret[field.key] && (values[field.key] ?? "").trim(),
      );
      if (fieldsToSave.length === 0) return Promise.resolve();

      return Promise.allSettled(
        fieldsToSave.map((field) =>
          SecretsService.createSecret(
            field.key,
            values[field.key].trim(),
            field.label,
          ),
        ),
      ).then((results) => {
        const saved = fieldsToSave
          .filter((_, i) => results[i].status === "fulfilled")
          .map((f) => f.key);
        const failed = fieldsToSave
          .filter((_, i) => results[i].status === "rejected")
          .map((f) => f.key);

        if (saved.length > 0) {
          // Refresh any cached secrets lists so a newly-saved secret shows up
          // in Settings → Secrets immediately. Without this, the 5-minute
          // staleTime on the secrets query (use-get-secrets.ts) can keep a
          // previously-loaded list stale and hide the new secret. Mirrors the
          // invalidation done by secret-form.tsx and secrets-settings.tsx.
          queryClient.invalidateQueries({ queryKey: ["secrets-search"] });
          queryClient.invalidateQueries({ queryKey: ["secrets"] });

          displaySuccessToast(
            t(I18nKey.MCP$SECRETS_SAVED, { keys: formatKeyList(saved) }),
          );
        }
        if (failed.length > 0) {
          displayErrorToast(
            t(I18nKey.MCP$SECRETS_SAVE_FAILED, { keys: formatKeyList(failed) }),
          );
        }
      });
    },
    [t, queryClient],
  );
}
