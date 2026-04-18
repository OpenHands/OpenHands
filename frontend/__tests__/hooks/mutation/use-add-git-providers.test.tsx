import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsService } from "#/api/secrets-service";
import { useAddGitProviders } from "#/hooks/mutation/use-add-git-providers";
import { Provider, ProviderToken } from "#/types/settings";

const mockTrackGitProviderConnected = vi.fn();

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackGitProviderConnected: mockTrackGitProviderConnected,
  }),
}));

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: "org-1" }),
}));

const buildProviders = (
  overrides: Partial<Record<Provider, ProviderToken>> = {},
): Record<Provider, ProviderToken> => ({
  github: { token: "", host: null },
  gitlab: { token: "", host: null },
  bitbucket: { token: "", host: null },
  bitbucket_data_center: { token: "", host: null },
  azure_devops: { token: "", host: null },
  forgejo: { token: "", host: null },
  enterprise_sso: { token: "", host: null },
  ...overrides,
});

describe("useAddGitProviders", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockTrackGitProviderConnected.mockReset();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  it("invalidates personal settings queries after saving providers", async () => {
    vi.spyOn(SecretsService, "addGitProvider").mockResolvedValue(true);

    const personalSettingsQueryKey = ["settings", "personal", "org-1"] as const;
    queryClient.setQueryData(personalSettingsQueryKey, {
      provider_tokens_set: {},
    });

    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useAddGitProviders(), { wrapper });

    await result.current.mutateAsync({
      providers: buildProviders({
        github: { token: "ghp_test_123", host: null },
      }),
    });

    expect(invalidateSpy).toHaveBeenCalled();
    expect(queryClient.getQueryState(personalSettingsQueryKey)?.isInvalidated).toBe(
      true,
    );
  });
});
