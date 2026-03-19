import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useSettings } from "#/hooks/query/use-settings";
import { useGetSecrets } from "#/hooks/query/use-get-secrets";
import { useApiKeys } from "#/hooks/query/use-api-keys";
import SettingsService from "#/api/settings-service/settings-service.api";
import { SecretsService } from "#/api/secrets-service";
import ApiKeysClient from "#/api/api-keys";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { app_mode: "saas" },
  }),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({
    data: true,
  }),
}));

vi.mock("#/hooks/use-is-on-intermediate-page", () => ({
  useIsOnIntermediatePage: () => false,
}));

describe("Organization-scoped query hooks", () => {
  let queryClient: QueryClient;

  const createWrapper = () => {
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    useSelectedOrganizationStore.setState({ organizationId: "org-1" });
    vi.clearAllMocks();
  });

  describe("useSettings", () => {
    it("should include organizationId in query key for proper cache isolation", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

      const { result } = renderHook(() => useSettings(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isFetched).toBe(true));

      // Verify the query was cached with the org-specific key
      const cachedData = queryClient.getQueryData(["settings", "org-1"]);
      expect(cachedData).toBeDefined();

      // Verify no data is cached under the old key without org ID
      const oldKeyData = queryClient.getQueryData(["settings"]);
      expect(oldKeyData).toBeUndefined();
    });

    it("should refetch when organization changes", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        language: "en",
      });

      // First render with org-1
      const { result, rerender } = renderHook(() => useSettings(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isFetched).toBe(true));
      expect(getSettingsSpy).toHaveBeenCalledTimes(1);

      // Change organization
      useSelectedOrganizationStore.setState({ organizationId: "org-2" });
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        language: "es",
      });

      // Rerender to pick up the new org ID
      rerender();

      await waitFor(() => {
        // Should have fetched again for the new org
        expect(getSettingsSpy).toHaveBeenCalledTimes(2);
      });

      // Verify both org caches exist independently
      const org1Data = queryClient.getQueryData(["settings", "org-1"]);
      const org2Data = queryClient.getQueryData(["settings", "org-2"]);
      expect(org1Data).toBeDefined();
      expect(org2Data).toBeDefined();
    });
  });

  describe("useGetSecrets", () => {
    it("should include organizationId in query key for proper cache isolation", async () => {
      const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");
      getSecretsSpy.mockResolvedValue([]);

      const { result } = renderHook(() => useGetSecrets(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isFetched).toBe(true));

      // Verify the query was cached with the org-specific key
      const cachedData = queryClient.getQueryData(["secrets", "org-1"]);
      expect(cachedData).toBeDefined();

      // Verify no data is cached under the old key without org ID
      const oldKeyData = queryClient.getQueryData(["secrets"]);
      expect(oldKeyData).toBeUndefined();
    });

    it("should fetch different data when organization changes", async () => {
      const getSecretsSpy = vi.spyOn(SecretsService, "getSecrets");

      // Mock different secrets for different orgs
      getSecretsSpy.mockResolvedValueOnce([
        { name: "SECRET_ORG_1", description: "Org 1 secret" },
      ]);

      const { result, rerender } = renderHook(() => useGetSecrets(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isFetched).toBe(true));
      expect(result.current.data).toHaveLength(1);
      expect(result.current.data?.[0].name).toBe("SECRET_ORG_1");

      // Change organization
      useSelectedOrganizationStore.setState({ organizationId: "org-2" });
      getSecretsSpy.mockResolvedValueOnce([
        { name: "SECRET_ORG_2", description: "Org 2 secret" },
      ]);

      rerender();

      await waitFor(() => {
        expect(result.current.data?.[0]?.name).toBe("SECRET_ORG_2");
      });
    });
  });

  describe("useApiKeys", () => {
    it("should include organizationId in query key for proper cache isolation", async () => {
      const getApiKeysSpy = vi.spyOn(ApiKeysClient, "getApiKeys");
      getApiKeysSpy.mockResolvedValue([]);

      const { result } = renderHook(() => useApiKeys(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isFetched).toBe(true));

      // Verify the query was cached with the org-specific key
      const cachedData = queryClient.getQueryData(["api-keys", "org-1"]);
      expect(cachedData).toBeDefined();

      // Verify no data is cached under the old key without org ID
      const oldKeyData = queryClient.getQueryData(["api-keys"]);
      expect(oldKeyData).toBeUndefined();
    });
  });

  describe("Cache isolation between organizations", () => {
    it("should maintain separate caches for each organization", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

      // Simulate fetching for org-1
      getSettingsSpy.mockResolvedValueOnce({
        ...MOCK_DEFAULT_USER_SETTINGS,
        language: "en",
      });

      useSelectedOrganizationStore.setState({ organizationId: "org-1" });
      const { rerender } = renderHook(() => useSettings(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(queryClient.getQueryData(["settings", "org-1"])).toBeDefined();
      });

      // Switch to org-2
      getSettingsSpy.mockResolvedValueOnce({
        ...MOCK_DEFAULT_USER_SETTINGS,
        language: "fr",
      });

      useSelectedOrganizationStore.setState({ organizationId: "org-2" });
      rerender();

      await waitFor(() => {
        expect(queryClient.getQueryData(["settings", "org-2"])).toBeDefined();
      });

      // Switch back to org-1 - should use cached data, not refetch
      useSelectedOrganizationStore.setState({ organizationId: "org-1" });
      rerender();

      // org-1 data should still be in cache
      const org1Cache = queryClient.getQueryData(["settings", "org-1"]) as any;
      expect(org1Cache?.language).toBe("en");

      // org-2 data should also still be in cache
      const org2Cache = queryClient.getQueryData(["settings", "org-2"]) as any;
      expect(org2Cache?.language).toBe("fr");
    });
  });
});
