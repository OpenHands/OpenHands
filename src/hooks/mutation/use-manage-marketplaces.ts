import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import MarketplacesService from "#/api/marketplaces-service";
import { MARKETPLACES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import type {
  AddMarketplaceRequest,
  UpdateMarketplaceRequest,
} from "#/types/marketplaces";
import {
  displayApiErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

/** Register a marketplace; its catalog is fetched as part of the request. */
export function useAddMarketplace() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    meta: { disableToast: true },
    mutationFn: (request: AddMarketplaceRequest) =>
      MarketplacesService.add(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MARKETPLACES_QUERY_KEYS.all });
      displaySuccessToast(t(I18nKey.MARKETPLACES$ADD_SUCCESS));
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}

export function useUpdateMarketplace() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    meta: { disableToast: true },
    mutationFn: ({
      name,
      request,
    }: {
      name: string;
      request: UpdateMarketplaceRequest;
    }) => MarketplacesService.update(name, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MARKETPLACES_QUERY_KEYS.all });
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}

export function useRemoveMarketplace() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    meta: { disableToast: true },
    mutationFn: (name: string) => MarketplacesService.remove(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MARKETPLACES_QUERY_KEYS.all });
      displaySuccessToast(t(I18nKey.MARKETPLACES$REMOVE_SUCCESS));
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}

/** Re-fetch the marketplace; installs that track its ref may gain an update. */
export function useRefreshMarketplace() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");

  return useMutation({
    meta: { disableToast: true },
    mutationFn: (name: string) => MarketplacesService.refresh(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MARKETPLACES_QUERY_KEYS.all });
      displaySuccessToast(t(I18nKey.MARKETPLACES$REFRESH_SUCCESS));
    },
    onError: (error) => displayApiErrorToast(error, t(I18nKey.ERROR$GENERIC)),
  });
}
