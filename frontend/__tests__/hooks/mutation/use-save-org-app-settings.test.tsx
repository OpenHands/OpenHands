import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ReactNode } from "react";
import { useSaveOrgAppSettings } from "#/hooks/mutation/use-save-org-app-settings";
import { organizationService } from "#/api/organization-service/organization-service.api";

vi.mock("#/api/organization-service/organization-service.api", () => ({
  organizationService: {
    saveOrganizationAppSettings: vi.fn(),
  },
}));

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: vi.fn(),
}));

import { useSelectedOrganizationId } from "#/context/use-selected-organization";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockResponse = {
  enable_proactive_conversation_starters: true,
  max_budget_per_task: 100,
  registered_marketplaces: [
    {
      name: "test-marketplace",
      source: "github:owner/repo",
      auto_load: true,
      scope: "org" as const,
    },
  ],
  updated_at: "2024-01-01T00:00:00Z",
};

describe("useSaveOrgAppSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("saves settings using the selected org from context", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    vi.mocked(organizationService.saveOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      registered_marketplaces: [
        { name: "test", source: "github:owner/repo", scope: "org" as const },
      ],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(
      organizationService.saveOrganizationAppSettings,
    ).toHaveBeenCalledWith({
      registered_marketplaces: [{ name: "test", source: "github:owner/repo", scope: "org" as const }],
    });
  });

  it("handles save error", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    const error = new Error("Failed to save");
    vi.mocked(organizationService.saveOrganizationAppSettings).mockRejectedValue(
      error,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({});

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBe(error);
  });

  it("uses mutateAsync for promise-based save", async () => {
    vi.mocked(useSelectedOrganizationId).mockReturnValue({ organizationId: "org-123", setOrganizationId: vi.fn() });
    vi.mocked(organizationService.saveOrganizationAppSettings).mockResolvedValue(
      mockResponse,
    );

    const { result } = renderHook(() => useSaveOrgAppSettings(), {
      wrapper: createWrapper(),
    });

    const promise = result.current.mutateAsync({
      max_budget_per_task: 50,
    });

    await expect(promise).resolves.toEqual(mockResponse);

    expect(
      organizationService.saveOrganizationAppSettings,
    ).toHaveBeenCalledWith({
      max_budget_per_task: 50,
    });
  });
});
