import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useSettings } from "#/hooks/query/use-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import type { ApiSettings } from "#/types/settings";

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

describe("useSettings", () => {
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

  it("should normalize a null llm_base_url from the API to an empty string", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    const apiSettings: ApiSettings = {
      ...MOCK_DEFAULT_USER_SETTINGS,
      llm_base_url: null,
    };
    getSettingsSpy.mockResolvedValue(apiSettings);

    const { result } = renderHook(() => useSettings(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isFetched).toBe(true));

    expect(result.current.data?.llm_base_url).toBe("");
  });
});
