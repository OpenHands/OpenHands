import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsService } from "#/api/secrets-service";
import { useDeleteGitProviders } from "#/hooks/mutation/use-delete-git-providers";

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: "org-1" }),
}));

describe("useDeleteGitProviders", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.restoreAllMocks();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
  });

  it("invalidates personal settings queries after deleting providers", async () => {
    vi.spyOn(SecretsService, "deleteGitProviders").mockResolvedValue(true);

    const personalSettingsQueryKey = ["settings", "personal", "org-1"] as const;
    queryClient.setQueryData(personalSettingsQueryKey, {
      provider_tokens_set: { github: null },
    });

    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useDeleteGitProviders(), { wrapper });

    await result.current.mutateAsync();

    expect(invalidateSpy).toHaveBeenCalled();
    expect(queryClient.getQueryState(personalSettingsQueryKey)?.isInvalidated).toBe(
      true,
    );
  });
});
