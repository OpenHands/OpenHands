import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import SettingsService from "#/api/settings-service/settings-service.api";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { MarketplaceRegistration } from "#/types/settings";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { I18nKey } from "#/i18n/declaration";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import {
  ORGANIZATION_APP_SETTINGS_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { useMarketplaceSkills } from "./use-get-marketplace-skills";

export interface MarketplaceMutationsOptions {
  onSuccess?: () => void;
  onSettled?: () => void;
}

export interface UseMarketplaceMutationsReturn {
  savePersonal: ReturnType<
    typeof useMutation<unknown, Error, MarketplaceRegistration[]>
  >;
  saveOrg: ReturnType<
    typeof useMutation<
      unknown,
      AxiosError,
      {
        marketplaces: MarketplaceRegistration[];
        lastKnownUpdatedAt: string | null;
      }
    >
  >;
  deletePersonal: ReturnType<typeof useMutation<unknown, Error, string>>;
  deleteOrg: ReturnType<
    typeof useMutation<
      unknown,
      AxiosError,
      { marketplaceName: string; lastKnownUpdatedAt: string | null }
    >
  >;
}

/**
 * Hook for marketplace CRUD operations.
 * Handles both personal (user settings) and org (app settings) marketplaces.
 * Follows repo pattern: onSuccess/onError callbacks, meta: { disableToast: true }
 */
export function useMarketplaceMutations(
  options: MarketplaceMutationsOptions = {},
): UseMarketplaceMutationsReturn {
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const { organizationId } = useSelectedOrganizationId();
  const marketplaceSkillsMutation = useMarketplaceSkills();

  // Save personal marketplace mutation
  const savePersonal = useMutation({
    mutationFn: async (marketplaces: MarketplaceRegistration[]) => {
      // Validate marketplace before saving
      if (marketplaces.length > 0) {
        const preview =
          await marketplaceSkillsMutation.mutateAsync(marketplaces);
        if (preview.errors?.length) {
          throw new Error(`Validation failed: ${preview.errors.join(", ")}`);
        }
      }
      // Backend will automatically set scope='personal' for user settings
      await SettingsService.saveSettings({
        registered_marketplaces: marketplaces,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ["skills"] });
      options.onSuccess?.();
    },
    onError: (error: Error) => {
      // Surface the backend detail (e.g. duplicate-name) for Axios errors;
      // fall back to the thrown message for local validation errors.
      const message =
        error instanceof AxiosError
          ? retrieveAxiosErrorMessage(error)
          : error.message;
      displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
    },
    onSettled: options.onSettled,
    meta: { disableToast: true },
  });

  // Save org marketplace mutation with 409 conflict handling
  const saveOrg = useMutation({
    mutationFn: async ({
      marketplaces,
      lastKnownUpdatedAt,
    }: {
      marketplaces: MarketplaceRegistration[];
      lastKnownUpdatedAt: string | null;
    }) => {
      // Validate marketplace before saving
      if (marketplaces.length > 0) {
        const preview =
          await marketplaceSkillsMutation.mutateAsync(marketplaces);
        if (preview.errors?.length) {
          throw new Error(`Validation failed: ${preview.errors.join(", ")}`);
        }
      }
      // Backend will automatically set scope='org' for org settings
      return organizationService.saveOrganizationAppSettings({
        registered_marketplaces: marketplaces,
        last_known_updated_at: lastKnownUpdatedAt,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      // Invalidate both org settings and user settings (which has inherited_marketplaces)
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
      });
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ["skills"] });
      options.onSuccess?.();
    },
    onError: (error: AxiosError) => {
      if (error.response?.status === 409) {
        // Concurrent modification - refetch and retry once
        queryClient.invalidateQueries({
          queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
        });
        queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
        displayErrorToast("Settings were modified. Please retry.");
      } else {
        displayErrorToast(
          retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
        );
      }
    },
    onSettled: options.onSettled,
    meta: { disableToast: true },
  });

  // Delete personal marketplace mutation
  // Fetches fresh state at mutation time to avoid stale closure issues
  const deletePersonal = useMutation({
    mutationFn: async (marketplaceName: string) => {
      const settings = await SettingsService.getSettings();
      // Name is the marketplace identity (unique across scopes).
      const updated = (settings.registered_marketplaces || []).filter(
        (mp) => mp.name !== marketplaceName,
      );
      // Backend will automatically set scope='personal' for user settings
      await SettingsService.saveSettings({
        registered_marketplaces: updated,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ["skills"] });
      options.onSuccess?.();
    },
    onError: (error: Error) => {
      const message =
        error instanceof AxiosError
          ? retrieveAxiosErrorMessage(error)
          : error.message;
      displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
    },
    onSettled: options.onSettled,
    meta: { disableToast: true },
  });

  // Delete org marketplace mutation with 409 conflict handling
  // Fetches fresh state at mutation time to avoid stale closure issues
  const deleteOrg = useMutation({
    mutationFn: async ({
      marketplaceName,
      lastKnownUpdatedAt,
    }: {
      marketplaceName: string;
      lastKnownUpdatedAt: string | null;
    }) => {
      const settings = await organizationService.getOrganizationAppSettings();
      // Name is the marketplace identity (unique across scopes).
      const updated = (settings.registered_marketplaces || []).filter(
        (mp) => mp.name !== marketplaceName,
      );
      // Backend will automatically set scope='org' for org settings
      return organizationService.saveOrganizationAppSettings({
        registered_marketplaces: updated,
        last_known_updated_at: lastKnownUpdatedAt,
      });
    },
    onSuccess: () => {
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      // Invalidate both org settings and user settings (which has inherited_marketplaces)
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
      });
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({ queryKey: ["skills"] });
      options.onSuccess?.();
    },
    onError: (error: AxiosError) => {
      if (error.response?.status === 409) {
        queryClient.invalidateQueries({
          queryKey: ORGANIZATION_APP_SETTINGS_KEYS.byOrg(organizationId),
        });
        queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
        displayErrorToast("Settings were modified. Please retry.");
      } else {
        displayErrorToast(
          retrieveAxiosErrorMessage(error) || t(I18nKey.ERROR$GENERIC),
        );
      }
    },
    onSettled: options.onSettled,
    meta: { disableToast: true },
  });

  return { savePersonal, saveOrg, deletePersonal, deleteOrg };
}
