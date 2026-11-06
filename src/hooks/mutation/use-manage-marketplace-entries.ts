import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import MarketplacesService from "#/api/marketplaces-service";
import { MARKETPLACES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import type { CatalogEntry } from "#/types/marketplaces";
import {
  displayApiErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

export type EntryAction = "install" | "update" | "uninstall" | "enable";

interface EntryActionInput {
  action: EntryAction;
  entry: CatalogEntry;
  /** Only for `enable`. */
  enabled?: boolean;
}

const SUCCESS_KEY: Record<Exclude<EntryAction, "enable">, I18nKey> = {
  install: I18nKey.MARKETPLACES$INSTALL_SUCCESS,
  update: I18nKey.MARKETPLACES$UPDATE_SUCCESS,
  uninstall: I18nKey.MARKETPLACES$UNINSTALL_SUCCESS,
};

function runAction({
  action,
  entry,
  enabled,
}: EntryActionInput): Promise<CatalogEntry> {
  switch (action) {
    case "install":
      return MarketplacesService.installEntry(entry.marketplace, entry.id);
    case "update":
      return MarketplacesService.updateEntry(entry.marketplace, entry.id);
    case "uninstall":
      return MarketplacesService.uninstallEntry(entry.marketplace, entry.id);
    case "enable":
      return MarketplacesService.setEntryEnabled(
        entry.marketplace,
        entry.id,
        enabled ?? true,
      );
    default:
      return Promise.reject(new Error(`Unknown entry action: ${action}`));
  }
}

/**
 * One mutation for every per-entry action on a marketplace page. The catalog
 * carries the install state, so every action invalidates the marketplace
 * queries rather than patching a cache.
 */
export function useManageMarketplaceEntry() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    meta: { disableToast: true },
    mutationFn: runAction,
    onSuccess: (_entry, variables) => {
      queryClient.invalidateQueries({ queryKey: MARKETPLACES_QUERY_KEYS.all });
      if (variables.action !== "enable") {
        displaySuccessToast(
          t(SUCCESS_KEY[variables.action], { name: variables.entry.name }),
        );
      }
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}
